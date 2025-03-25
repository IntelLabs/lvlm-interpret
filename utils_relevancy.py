import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

from transformers.utils import  logging
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from transformers.generation.beam_search import  BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers import (
    StoppingCriteriaList)
from transformers.generation.utils import GenerateOutput

import gradio as gr
from tqdm import tqdm

logger = logging.get_logger(__name__)

SEPARATORS_LIST = ['.',',','?','!', ':', ';', '</s>', '/', '!', '(', ')', '[', ']', '{', '}', '<', '>', '|', '\\', '-', '_', '+', '=', '*', '&', '^', '%', '$', '#', '@', '!', '~', '`', ' ', '\t', '\n', '\r', '\x0b', '\x0c']


# rule 5 from paper
def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam

# rule 6 from paper
def handle_self_attention_image(R_i_i, enc_attn_weights, privious_cam=[]):
    if privious_cam :
        device = privious_cam[-1].device
    else:
        device = None
    for i, blk in enumerate(enc_attn_weights):
        grad = blk.grad.float().detach()
        # if model.use_lrp: # not used
        #     cam = blk[batch_no].detach()
        # else:
        cam = blk.float().detach() # the attention of one layer
        if device is None:
            device = cam.device
        cam = avg_heads(cam.to(device), grad.to(device))
        # rebuild the privious attenions to the same size as the current attention
        if len(privious_cam) != 0 and cam.shape[0] == 1:
            len_seq, all_len_seq = privious_cam[i].shape
            assert len_seq == all_len_seq, "The privious CAMs are not square"
            new_column = torch.zeros(len_seq, 1).to(cam.device)
            privious_cam[i] = torch.cat((privious_cam[i], new_column), dim=1)
            privious_cam[i] = torch.cat((privious_cam[i], cam), dim=0)
            cam = privious_cam[i]
        elif cam.shape[0] != 1:
            privious_cam.append(cam)
        assert cam.shape == R_i_i.shape, "The attention weights and the relevancy map are not the same size"
        R_i_i += torch.matmul(cam, R_i_i)
        del grad, cam
        # torch.cuda.empty_cache()

    return R_i_i, privious_cam

def handle_self_attention_image_vit(R_i_i_init, enc_attn_weights_vit, img_idx=None, add_skip=False, normalize=False):
    if img_idx:
        R_i_i = R_i_i_init[img_idx:img_idx+576, img_idx:img_idx+576] 
        if add_skip:
            R_i_i = R_i_i + torch.eye(R_i_i.shape[-1]).to(R_i_i.device)
        # add a first column and first row of zeros to R_i_i - option #1
        R_i_i = torch.cat((torch.zeros(1, R_i_i.shape[1]).to(R_i_i.device), R_i_i), dim=0)
        R_i_i = torch.cat((torch.zeros(R_i_i.shape[0], 1).to(R_i_i.device), R_i_i), dim=1)
        R_i_i[0,0] = 1
    else:
        R_i_i = R_i_i_init
    if normalize:
        R_i_i = handle_residual(R_i_i)
    for j, blk_vit in enumerate(enc_attn_weights_vit): #577x577, 1x576
        grad_vit = blk_vit.grad.float().detach()
        cam_vit = blk_vit.float().detach()
        cam_vit = avg_heads(cam_vit, grad_vit)
        assert cam_vit.shape == R_i_i.shape, "The vit relevancy map and the llama relevancy map are not the same size"
        R_i_i += torch.matmul(cam_vit, R_i_i)
    return R_i_i

def compute_rollout_attention(all_layer_matrices_raw, start_layer=0, average_positive=False, add_residual=False):
    all_layer_matrices = []
    # image average self attention in the encoder
    for blk in all_layer_matrices_raw:
        cam = blk.squeeze().detach() #16x577x577
        if average_positive:
            cam = cam.clamp(min=0).mean(dim=0)
        else:
            cam = cam.mean(dim=0)
        all_layer_matrices.append(cam) #577x577
    layer_attn_avg = [all_layer_matrices[i].detach().clone() for i in range(len(all_layer_matrices))]
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[-1]
    eye = torch.eye(num_tokens).to(all_layer_matrices[0].device) #577x577
    if add_residual == "start":
        all_layer_matrices[start_layer] = eye + all_layer_matrices[start_layer]
        all_layer_matrices[start_layer] = all_layer_matrices[start_layer] / all_layer_matrices[start_layer].sum(dim=-1, keepdim=True)
    elif add_residual == "all":
        all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
        all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                            for i in range(len(all_layer_matrices))]
    
    matrices_aug = all_layer_matrices
    joint_attention = matrices_aug[start_layer]
    if start_layer == 0:
        for i in range(start_layer+1, len(matrices_aug)):
            joint_attention = matrices_aug[i].matmul(joint_attention)
    if start_layer == len(matrices_aug)-1:
        for i in range(start_layer-1, -1,-1):
            joint_attention = matrices_aug[i].matmul(joint_attention)
    return joint_attention, layer_attn_avg

# normalization- eq. 8+9
def handle_residual(orig_self_attention):
    self_attention = orig_self_attention.clone()
    diag_idx = range(self_attention.shape[-1])
    self_attention -= torch.eye(self_attention.shape[-1]).to(self_attention.device)
    assert self_attention[diag_idx, diag_idx].min() >= 0
    sum_rows = self_attention.sum(dim=-1, keepdim=True)
    sum_rows[sum_rows == 0] = 1 # replace all elements equal to zero by 1
    self_attention = self_attention / sum_rows# this has nan elements ue to divoding by zero
    self_attention += torch.eye(self_attention.shape[-1]).to(self_attention.device)
    return self_attention

def compute_word_rel_map(tokens, target_index, R_i_i, separators_list, 
                         current_rel_map, current_count, current_word, word_rel_maps):
    if target_index == 0:
        current_word = tokens[target_index]
        current_rel_map = R_i_i
        current_count = 1
    # If the token is a part of the current word, add its relevancy map to the current word's relevancy map
    else:
        if not tokens[target_index].startswith('▁') and tokens[target_index] not in separators_list:
            current_word += tokens[target_index]
            # If current_rel_map is smaller, pad it with zeros
            if current_rel_map.shape[0] < R_i_i.shape[0]:
                # Calculate the padding sizes
                padding = (0, R_i_i.shape[1] - current_rel_map.shape[1], 0, R_i_i.shape[0] - current_rel_map.shape[0])
                # Pad rel_maps[1] with zeros
                current_rel_map = F.pad(current_rel_map, padding, "constant", 0)
            current_rel_map += R_i_i
            current_count += 1
        else:
            # Otherwise, store the current word's relevancy map and start a new word
            word_rel_maps[current_word] = current_rel_map / current_count
            current_word = tokens[target_index]
            current_rel_map = R_i_i
            current_count = 1
    return word_rel_maps, current_rel_map, current_count, current_word


def construct_relevancy_map(tokenizer, model, input_ids, tokens, outputs, output_ids, img_idx, apply_normalization=True, progress=gr.Progress(track_tqdm=True)):
    logger.debug('Tokens: %s', tokens)
    enable_vit_relevancy = len(model.enc_attn_weights_vit) > 0
    if enable_vit_relevancy:
        enc_attn_weights_vit = model.enc_attn_weights_vit
    enc_attn_weights = model.enc_attn_weights
    device = outputs.attentions[-1][0][0].device

    # compute rollout attention
    # start_layer = len(enc_attn_weights_vit)-2 # the last layer is not considered for llava
    # rollout_vit, layer_attn_avg = compute_rollout_attention(enc_attn_weights_vit, start_layer,average_positive=False, add_residual=False)

    # compute relevancy maps
    rel_maps = []
    rel_maps_all = []
    rel_maps_vit = []
    rel_maps_all_generated_token = []

    num_generated_tokens = len(outputs.attentions)
    num_self_att_layers = len(outputs.attentions[0])
    assert num_generated_tokens == len(outputs.scores)
    assert num_generated_tokens*num_self_att_layers == len(enc_attn_weights), f'{num_generated_tokens}x{num_self_att_layers} != {len(enc_attn_weights)}'
    # rearenge the attention weights the same as outputs.attentions
    enc_attn_weights = [enc_attn_weights[i*num_self_att_layers : (i+1)*num_self_att_layers] for i in range(num_generated_tokens)]


    assert len(tokens) == len(outputs.scores), f'Length of tokens {len(tokens)} is not equal to the length of outputs.scores {len(outputs.scores)}\ntokens: {tokens}'
    clean_tokens = []

    # Initialize dictionaries
    word_rel_maps_llama, word_rel_maps_all, word_rel_maps_vit, word_rel_maps_all_generated_token = {}, {}, {}, {}
    word_counts = {}

    # Initialize the averaged attention map for the first token
    privious_cam = []

    # Initialize current_rel_map and current_word variables
    current_rel_map, current_rel_map_all, current_rel_map_all_generated_token, current_rel_map_vit = None, None, None, None
    current_word, current_word_all, current_word_vit, current_word_all_generated_token = None, None, None, None

    # Initialize current_count variables
    current_count, current_count_vit, current_count_all, current_count_all_generated_token = 0, 0, 0, 0

    if enable_vit_relevancy:
        enc_attn_weights_vit = enc_attn_weights_vit[:-1] # last layer is not considered for llava
        assert len(enc_attn_weights_vit) > 0

    rel_maps_dict = {}
    logger.debug(f'Number of output scores: {len(outputs.scores)}')
    for target_index in tqdm(range(len(outputs.scores)), desc="Building relevancy maps"): #the last token is </s>
        clean_tokens.append(tokens[target_index])
        token_logits = outputs.scores[target_index]
        token_id = torch.tensor(output_ids[target_index]).to(device)

        # print out the token and its predicted id
        #print(f'Token: {tokens[target_index]}, Predicted ID: {token_id}')
        if token_id != output_ids[target_index]:
            logger.warning(f'The token_id_max_score is not the same as the output_id')
            # print the decoded token
            logger.warning(f'Decoded Token: {tokenizer.decode(token_id)}')
            logger.warning(f'The generated output: {tokens[target_index]}')
        # check if the output_id is the same as the token_id
        assert token_id == output_ids[target_index], "The token_id_max_score is not the same as the output_id"
        

        token_id_one_hot = torch.nn.functional.one_hot(token_id, num_classes=token_logits.size(-1)).float()
        token_id_one_hot = token_id_one_hot.view(1, -1)
        token_id_one_hot.requires_grad_(True)

        # Compute loss and backpropagate to get gradients on attention weights
        model.zero_grad()
        token_logits.backward(gradient=token_id_one_hot, retain_graph=True)
    
        # initialize relevancy map for llama
        R_i_i_init = torch.eye(enc_attn_weights[target_index][0].shape[-1], enc_attn_weights[target_index][0].shape[-1]).to(token_logits.device).float()
        # compute relevancy map accourding to rule #6
        R_i_i, privious_cam = handle_self_attention_image(R_i_i_init, enc_attn_weights[target_index], privious_cam)

        if enable_vit_relevancy:
            # initialize the vit relevancy map with the llama relevancy map
            R_i_i_all = handle_self_attention_image_vit(R_i_i, enc_attn_weights_vit, img_idx, add_skip=False, normalize=False)
            
            # initialize using the relevancy map of the generated token to the image - option #1
            R_i_i_init_vit_all = torch.eye(enc_attn_weights_vit[0].shape[-1], enc_attn_weights_vit[0].shape[-1]).to(token_logits.device).float()
            
            # add R_i_i[-1,:][img_idx:img_idx+576] to the first row and column of R_i_i_init_vit - option #2
            R_i_i_init_vit_all[0,1:] = R_i_i_init_vit_all[0,1:] + R_i_i[-1,:][img_idx:img_idx+576]
            R_i_i_init_vit_all[1:,0] = R_i_i_init_vit_all[1:,0] + R_i_i[-1,:][img_idx:img_idx+576]
            R_i_i_all_generated_token = handle_self_attention_image_vit(R_i_i_init_vit_all, enc_attn_weights_vit)
            
            # compute ViT relevancy map
            R_i_i_init_vit = torch.eye(enc_attn_weights_vit[0].shape[-1], enc_attn_weights_vit[0].shape[-1]).to(token_logits.device).float()
            R_i_i_vit = handle_self_attention_image_vit(R_i_i_init_vit, enc_attn_weights_vit)
        if apply_normalization:
            R_i_i = handle_residual(R_i_i)
            if enable_vit_relevancy:
                R_i_i_all = handle_residual(R_i_i_all)
                R_i_i_vit = handle_residual(R_i_i_vit)
                R_i_i_all_generated_token = handle_residual(R_i_i_all_generated_token)
        else:
            R_i_i = R_i_i - torch.eye(enc_attn_weights[target_index][0].shape[-1], enc_attn_weights[target_index][0].shape[-1]).to(token_logits.device).float()
        
        rel_maps.append(R_i_i)
        if enable_vit_relevancy:
            rel_maps_all.append(R_i_i_all)
            rel_maps_vit.append(R_i_i_vit)
            rel_maps_all_generated_token.append(R_i_i_all_generated_token)

        # values should be rel_maps, and the keys should be the tokens
        # check if this token already exsits
        if tokens[target_index] in rel_maps_dict.keys():
            tokens[target_index] = tokens[target_index] + '_'
        rel_maps_dict[tokens[target_index]] = R_i_i

        # If the token is a part of the current word, add its relevancy map to the current word's relevancy map
        word_rel_maps_llama, current_rel_map, current_count, current_word = compute_word_rel_map(
            tokens, target_index, R_i_i, SEPARATORS_LIST, 
            current_rel_map, current_count, current_word, word_rel_maps_llama)
        
        if enable_vit_relevancy:
            word_rel_maps_all, current_rel_map_all, current_count_all, current_word_all = compute_word_rel_map(
                tokens, target_index, R_i_i_all, SEPARATORS_LIST, 
                current_rel_map_all, current_count_all, current_word_all, word_rel_maps_all)
            
            word_rel_maps_vit, current_rel_map_vit, current_count_vit, current_word_vit = compute_word_rel_map(
                tokens, target_index, R_i_i_vit, SEPARATORS_LIST, 
                current_rel_map_vit, current_count_vit, current_word_vit, word_rel_maps_vit)
            
            word_rel_maps_all_generated_token, current_rel_map_all_generated_token, \
                current_count_all_generated_token, current_word_all_generated_token = compute_word_rel_map(
                    tokens, target_index, R_i_i_all_generated_token, SEPARATORS_LIST, 
                    current_rel_map_all_generated_token, current_count_all_generated_token, 
                    current_word_all_generated_token, word_rel_maps_all_generated_token
            )

        logger.debug(f'Current word: {current_word}')
        
    # Store the last word's relevancy map
    word_rel_maps_llama[current_word] = current_rel_map / current_count

    if enable_vit_relevancy:
        word_rel_maps_all[current_word_all] = current_rel_map_all / current_count_all
        word_rel_maps_vit[current_word_vit] = current_rel_map_vit / current_count_vit
        word_rel_maps_all_generated_token[current_word_all_generated_token] = current_rel_map_all_generated_token / current_count_all_generated_token


    word_rel_maps = {
        "llama": word_rel_maps_llama,
        "llama_token":rel_maps_dict,
        "vit": word_rel_maps_vit,
        "all": word_rel_maps_all,
        "all_v2": word_rel_maps_all_generated_token
    }

    return word_rel_maps

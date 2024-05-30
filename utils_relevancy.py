import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

from transformers.deepspeed import is_deepspeed_zero3_enabled
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

from tqdm import tqdm

logger = logging.get_logger(__name__)

IMAGE_TOKEN_INDEX = -200
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


from transformers.generation.utils import GenerateNonBeamOutput, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput, NEED_SETUP_CACHE_CLASSES_MAPPING
from transformers.generation.configuration_utils import GenerationMode


@torch.enable_grad()
def generate_with_grads(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    synced_gpus: Optional[bool] = None,
    assistant_model: Optional["PreTrainedModel"] = None,
    streamer: Optional["BaseStreamer"] = None,
    negative_prompt_ids: Optional[torch.Tensor] = None,
    negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> Union[GenerateOutput, torch.LongTensor]:
    r"""

    Generates sequences of token ids for models with a language modeling head.

    <Tip warning={true}>

    Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
    model's default generation configuration. You can override any `generation_config` by passing the corresponding
    parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

    For an overview of generation strategies and code examples, check out the [following
    guide](../generation_strategies).

    </Tip>

    Parameters:
        inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
            The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
            method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
            should be in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
            `input_ids`, `input_values`, `input_features`, or `pixel_values`.
        generation_config (`~generation.GenerationConfig`, *optional*):
            The generation configuration to be used as base parametrization for the generation call. `**kwargs`
            passed to generate matching the attributes of `generation_config` will override them. If
            `generation_config` is not provided, the default will be used, which has the following loading
            priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
            configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
            default values, whose documentation should be checked to parameterize generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            Custom logits processors that complement the default logits processors built from arguments and
            generation config. If a logit processor is passed that is already created with the arguments or a
            generation config an error is thrown. This feature is intended for advanced users.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            Custom stopping criteria that complements the default stopping criteria built from arguments and a
            generation config. If a stopping criteria is passed that is already created with the arguments or a
            generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
            sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
            intended for advanced users.
        prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
            If provided, this function constraints the beam search to allowed tokens only at each step. If not
            provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
            `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
            on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
            for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
            Retrieval](https://arxiv.org/abs/2010.00904).
        synced_gpus (`bool`, *optional*):
            Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
            `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
            generating before other GPUs. Otherwise it'll be set to `False`.
        assistant_model (`PreTrainedModel`, *optional*):
            An assistant model that can be used to accelerate generation. The assistant model must have the exact
            same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
            is much faster than running generation with the model you're calling generate from. As such, the
            assistant model should be much smaller.
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            The negative prompt needed for some processors such as CFG. The batch size must match the input batch
            size. This is an experimental feature, subject to breaking API changes in future versions.
        negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Attention_mask for `negative_prompt_ids`.
        kwargs (`Dict[str, Any]`, *optional*):
            Ad hoc parametrization of `generation_config` and/or additional model-specific kwargs that will be
            forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
            specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

    Return:
        [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
        or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

            If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateDecoderOnlyOutput`],
                - [`~generation.GenerateBeamDecoderOnlyOutput`]

            If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
            [`~utils.ModelOutput`] types are:

                - [`~generation.GenerateEncoderDecoderOutput`],
                - [`~generation.GenerateBeamEncoderDecoderOutput`]
    """
    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    self._validate_model_class()
    generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
    self._validate_model_kwargs(model_kwargs.copy())

    # 2. Set generation parameters if not already defined
    if synced_gpus is None:
        if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
            synced_gpus = True
        else:
            synced_gpus = False
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
        if model_kwargs.get("attention_mask", None) is None:
            logger.warning(
                "The attention mask and the pad token id were not set. As a consequence, you may observe "
                "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
            )
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
        generation_config.pad_token_id = eos_token_id

    # 3. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]

    # 4. Define other model kwargs
    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
    # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    # generating the first new token or not, and we only want to use the embeddings for the first new token)
    if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
        model_kwargs["use_cache"] = True
    else:
        model_kwargs["use_cache"] = generation_config.use_cache

    accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    requires_attention_mask = "encoder_outputs" not in model_kwargs

    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
        model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
        )

    # decoder-only models should use left-padding for generation
    if not self.config.is_encoder_decoder:
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
        if (
            generation_config.pad_token_id is not None
            and len(inputs_tensor.shape) == 2
            and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

    if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
        # if model is encoder decoder encoder_outputs are created
        # and added to `model_kwargs`
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    if self.config.is_encoder_decoder:
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
            batch_size=batch_size,
            model_input_name=model_input_name,
            model_kwargs=model_kwargs,
            decoder_start_token_id=generation_config.decoder_start_token_id,
            bos_token_id=generation_config.bos_token_id,
            device=inputs_tensor.device,
        )
    else:
        input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    if streamer is not None:
        streamer.put(input_ids.cpu())

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    if generation_config.max_new_tokens is not None:
        if not has_default_max_length and generation_config.max_length is not None:
            logger.warning(
                f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                "Please refer to the documentation for more information. "
                "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
            )
        generation_config.max_length = generation_config.max_new_tokens + input_ids_length

    # otherwise the total length [inputs-embeds-len + new-tokens-len] will go beyond indicated `max_length``
    elif (
        model_input_name == "inputs_embeds"
        and inputs_tensor.shape[:-1] != input_ids.shape
        and not self.config.is_encoder_decoder
    ):
        generation_config.max_length -= inputs_tensor.shape[1]
        generation_config.min_length = max(generation_config.min_length - inputs_tensor.shape[1], 0)

    if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
        if generation_config.cache_implementation == "static":
            if model_kwargs.get("past_key_values", False) is not False:
                raise ValueError(
                    "Using `past_key_values` argument with `generate()` when using a static KV cache is not supported. Please open an issue in Transformers GitHub repository."
                )
            cache_cls = NEED_SETUP_CACHE_CLASSES_MAPPING["static"]
            if not callable(getattr(self, "_setup_cache", None)):
                raise ValueError(
                    "The `generation_config` defines a `cache_implementation` that is not compatible with this model."
                    " Make sure it has a `_setup_cache` function."
                )
            self._setup_cache(cache_cls, max_batch_size=batch_size, max_cache_len=generation_config.max_length)

    self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    # 7. determine generation mode
    generation_mode = generation_config.get_generation_mode(assistant_model)

    if streamer is not None and (generation_config.num_beams > 1):
        raise ValueError(
            "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
        )

    if self.device.type != input_ids.device.type:
        warnings.warn(
            "You are calling .generate() with the `input_ids` being on a device type different"
            f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
            f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
            " Please make sure that you have put `input_ids` to the"
            f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
            " running `.generate()`.",
            UserWarning,
        )

    # 8. prepare distribution pre_processing samplers
    prepared_logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
        model_kwargs=model_kwargs,
        negative_prompt_ids=negative_prompt_ids,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
    )

    # 9. prepare stopping criteria
    prepared_stopping_criteria = self._get_stopping_criteria(
        generation_config=generation_config, stopping_criteria=stopping_criteria
    )
    # 10. go into different generation modes
    if generation_mode == GenerationMode.ASSISTED_GENERATION:
        if generation_config.num_return_sequences > 1:
            raise ValueError(
                "num_return_sequences has to be 1 when doing assisted generate, "
                f"but is {generation_config.num_return_sequences}."
            )
        if batch_size > 1:
            raise ValueError("assisted generate is only supported for batch_size = 1")
        if not model_kwargs["use_cache"]:
            raise ValueError("assisted generate requires `use_cache=True`")

        # 11. Get the candidate generator, given the parameterization
        candidate_generator = self._get_candidate_generator(
            generation_config=generation_config,
            input_ids=input_ids,
            inputs_tensor=inputs_tensor,
            assistant_model=assistant_model,
            logits_processor=logits_processor,
            model_kwargs=model_kwargs,
        )

        # 12. run assisted generate
        result = self.assisted_decoding(
            input_ids,
            candidate_generator=candidate_generator,
            do_sample=generation_config.do_sample,
            logits_processor=prepared_logits_processor,
            logits_warper=self._get_logits_warper(generation_config) if generation_config.do_sample else None,
            stopping_criteria=prepared_stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            output_logits=generation_config.output_logits,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )
    if generation_mode == GenerationMode.GREEDY_SEARCH:
        # 11. run greedy search
        result = self._greedy_search(
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            output_logits=generation_config.output_logits,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
        if not model_kwargs["use_cache"]:
            raise ValueError("Contrastive search requires `use_cache=True`")

        result = self._contrastive_search(
            input_ids,
            top_k=generation_config.top_k,
            penalty_alpha=generation_config.penalty_alpha,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            output_logits=generation_config.output_logits,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            sequential=generation_config.low_memory,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.SAMPLE:
        # 11. prepare logits warper
        logits_warper = self._get_logits_warper(generation_config)

        # 12. expand input_ids with `num_return_sequences` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 13. run sample
        result = self._sample(
            input_ids,
            logits_processor=prepared_logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=prepared_stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            output_logits=generation_config.output_logits,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            streamer=streamer,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.BEAM_SEARCH:
        # 11. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )
        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        # 13. run beam search
        result = self._beam_search(
            input_ids,
            beam_scorer,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            output_logits=generation_config.output_logits,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            sequential=generation_config.low_memory,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.BEAM_SAMPLE:
        # 11. prepare logits warper
        logits_warper = self._get_logits_warper(generation_config)

        # 12. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )

        # 13. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        # 14. run beam sample
        result = self._beam_sample(
            input_ids,
            beam_scorer,
            logits_processor=prepared_logits_processor,
            logits_warper=logits_warper,
            stopping_criteria=prepared_stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            output_logits=generation_config.output_logits,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
        # 11. prepare beam search scorer
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            num_beam_groups=generation_config.num_beam_groups,
            max_length=generation_config.max_length,
        )
        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        # 13. run beam search
        result = self._group_beam_search(
            input_ids,
            beam_scorer,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            output_logits=generation_config.output_logits,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
        final_constraints = []
        if generation_config.constraints is not None:
            final_constraints = generation_config.constraints

        if generation_config.force_words_ids is not None:

            def typeerror():
                raise ValueError(
                    "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
                    f"of positive integers, but is {generation_config.force_words_ids}."
                )

            if (
                not isinstance(generation_config.force_words_ids, list)
                or len(generation_config.force_words_ids) == 0
            ):
                typeerror()

            for word_ids in generation_config.force_words_ids:
                if isinstance(word_ids[0], list):
                    if not isinstance(word_ids, list) or len(word_ids) == 0:
                        typeerror()
                    if any(not isinstance(token_ids, list) for token_ids in word_ids):
                        typeerror()
                    if any(
                        any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                        for token_ids in word_ids
                    ):
                        typeerror()

                    constraint = DisjunctiveConstraint(word_ids)
                else:
                    if not isinstance(word_ids, list) or len(word_ids) == 0:
                        typeerror()
                    if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                        typeerror()

                    constraint = PhrasalConstraint(word_ids)
                final_constraints.append(constraint)

        # 11. prepare beam search scorer
        constrained_beam_scorer = ConstrainedBeamSearchScorer(
            constraints=final_constraints,
            batch_size=batch_size,
            num_beams=generation_config.num_beams,
            device=inputs_tensor.device,
            length_penalty=generation_config.length_penalty,
            do_early_stopping=generation_config.early_stopping,
            num_beam_hyps_to_keep=generation_config.num_return_sequences,
            max_length=generation_config.max_length,
        )
        # 12. interleave input_ids with `num_beams` additional sequences per batch
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=generation_config.num_beams,
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )
        # 13. run beam search
        result = self._constrained_beam_search(
            input_ids,
            constrained_beam_scorer=constrained_beam_scorer,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            pad_token_id=generation_config.pad_token_id,
            eos_token_id=generation_config.eos_token_id,
            output_scores=generation_config.output_scores,
            output_logits=generation_config.output_logits,
            return_dict_in_generate=generation_config.return_dict_in_generate,
            synced_gpus=synced_gpus,
            **model_kwargs,
        )

    if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
        if not callable(getattr(self, "_reset_cache", None)):
            raise ValueError(
                "A `static_cache` was used to generate but there was a failure when trying to  release the cache. "
                " Make sure this model implements a `_reset_cache` function."
            )
        self._reset_cache()

    return result


def construct_relevancy_map(tokenizer, model, input_ids, tokens, outputs, output_ids, img_idx, apply_normalization=True):
    logger.debug('Tokens: %s', tokens)
    enable_vit_relevancy = False
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
    for target_index in range(len(outputs.scores)): #the last token is </s>
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
            # initiate the vit relevancy map with the llama relevancy map
            R_i_i_all = handle_self_attention_image_vit(R_i_i, enc_attn_weights_vit, img_idx, add_skip=False, normalize=False)
            
            # initiate using the relevancy map of the generated token to the image - option #1
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

        # values should be rel_masps, ans the keys should be the tokens
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
import sys
sys.path.append('causality_lab')

import logging
import os
import numpy as np
import gradio as gr
import torch
from PIL import ImageDraw, Image
from matplotlib import pyplot as plt
from plot_utils import draw_graph, draw_pds_tree
from causal_discovery_utils.cond_indep_tests import CondIndepParCorr

logger = logging.getLogger(__name__)

from utils_causal_discovery_fn import (
    get_relevant_image_tokens,
    tokens_analysis,
    create_explanation,
    copy_sub_graph,
    show_tokens_on_image,
    calculate_explanation_pvals,
    get_relevant_prompt_tokens,
    get_relevant_text_tokens,
    crop_token,
    get_expla_set_per_rad
)


def create_im_tokens_marks(orig_img, tokens_to_mark, weights=None, txt=None, txt_pos=None):
    im_1 = orig_img.copy()
    if weights is not None:
        im_heat = show_tokens_on_image(tokens_to_mark, im_1, weights)
    else:
        im_heat = show_tokens_on_image(tokens_to_mark, im_1)
    im_heat_edit = ImageDraw.Draw(im_heat)
    if isinstance(txt, str):
        if txt_pos is None:
            txt_pos = (10, 10)
        im_heat_edit.text(txt_pos, txt, fill=(255, 255, 255))
    im_heat = im_heat_edit._image
    return im_heat


def causality_update_dropdown(state):
    generated_text = state.output_ids_decoded
    choices = [ f'{i}_{tok}' for i,tok in enumerate(generated_text)]
    return state, gr.Dropdown(value=choices[0], interactive=True, scale=2, choices=choices)


def handle_causal_head(state, explainers_data, head_selection, class_token_txt):
    recovered_image = state.recovered_image
    first_im_token_idx = state.image_idx

    token_to_explain = explainers_data[0]
    head_id = head_selection
    explainer = explainers_data[1][head_id]
    if explainer is None:
        return [], None

    expla_set_per_rad = get_expla_set_per_rad(explainer.results[token_to_explain]['pds_tree'])
    max_depth = max(expla_set_per_rad.keys())
    im_heat_list = []
    im_tok_rel_idx = []
    for rad in range(1,max_depth+1):
        im_tok_rel_idx += [v-first_im_token_idx 
                           for v in expla_set_per_rad[rad] if v >= first_im_token_idx and v < (first_im_token_idx+576)]
        im_heat_list.append(
            create_im_tokens_marks(recovered_image, im_tok_rel_idx, txt='search radius: {rad}'.format(rad=rad))
        )
        

    # im_graph_list = []
    # for r in range(1, 5):
    #     expla_list = explainer.explain(token_to_explain, max_range=r)[0][0]
    #     nodes_set = set(expla_list)
    #     nodes_set.add(token_to_explain)
    #     subgraph = copy_sub_graph(explainer.graph, nodes_set)
    #     fig = draw_graph(subgraph, show=False)
    #     fig.canvas.draw()
    #     im_graph = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    #     plt.close()
    #     im_graph_list.append(im_graph)

    expla_list = explainers_data[2][head_id]
    node_labels = dict()
    for tok in expla_list:
        im_idx = tok - first_im_token_idx
        if im_idx < 0 or im_idx >= 576:  # if token is not image
            continue
        im_tok = crop_token(recovered_image, im_idx, pad=2)
        node_labels[tok] = im_tok.resize((45, 45))

    node_labels[token_to_explain] = class_token_txt.split('_')[1]
    
    nodes_set = set(expla_list)
    nodes_set.add(token_to_explain)
    fig = draw_pds_tree(explainer.results[token_to_explain]['pds_tree'], explainer.graph, node_labels=node_labels,
                        node_size_factor=1.4)
    if fig is None:
        fig = plt.figure()
    fig.canvas.draw()
    im_graph = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    plt.close()

    return im_heat_list, im_graph


def handle_causality(state, state_causal_explainers, token_to_explain, alpha_ext=None, att_th_ext=None):
    # ---***------***------***------***------***------***------***------***------***------***------***------***---
    # ---***--- Results' containers ---***---
    gallery_image_list = []
    gallery_graph_list = []
    gallery_bar_graphs = []
    
    # ---***------***------***------***------***------***------***------***------***------***------***------***---
    # ---***--- Generic app handling ---***---
    if not hasattr(state, 'attention_key'):
        return []
    
    # ---***------***------***------***------***------***------***------***------***------***------***------***---
    # ---***--- Load attention matrix ---***---
    fn_attention = state.attention_key + '_attn.pt'
    recovered_image = state.recovered_image
    first_im_token_idx = state.image_idx
    generated_text = state.output_ids_decoded

    if not os.path.exists(fn_attention):
        gr.Error('Attention file not found. Please re-run query.')
    else:
        attentions = torch.load(fn_attention)  # attentions is a tuple of length as the number of generated tokens.

    last_mh_attention = attentions[-1][-1]  # last generated token, last layer
    num_heads, _, attention_len = last_mh_attention[-1].shape
    full_attention = np.zeros((num_heads, attention_len, attention_len))

    last_mh_attention = attentions[0][-1]  # last layer's attention matrices before output generation
    attention_vals = last_mh_attention[0].detach().cpu().numpy()  # 0 is the index for the sample in the batch.
    d1 = attention_vals.shape[-1]
    full_attention[:, :d1, :d1] = attention_vals

    # create one full attention matrix that includes attention to generated tokens
    for gen_idx in range(1, len(generated_text)):
        last_mh_attention = attentions[gen_idx][-1]
        att_np = last_mh_attention[0].detach().cpu().numpy()
        full_attention[:, d1, :att_np.shape[-1]] = att_np[:,0,:]
        d1 += 1

    # Sizes:
    # Number of heads: {num_heads}, attention size: {attention_len}x{attention_len}

    # ---***------***------***------***------***------***------***------***------***------***------***------***---
    # ---***--- Hyper-parameters for causal discovery ---***---
    threshold = 1e-5  # alpha; threshold for p-value in conditional independence testing
    degrees_of_freedom = 128
    default_search_range = 3
    max_num_image_tokens = 50  # number of image-tokens to consider as 'observed'. Used for calculating head importance
    att_th = 0.01  # threshold for attention values. Below this value, the token is considered 'not-attented'
    search_range = default_search_range  # causal-explanation seach-distance in the causal graph
    if alpha_ext is not None:
        threshold = alpha_ext
    if att_th_ext is not None:
        att_th = att_th_ext

    heads_to_analyse = list(range(num_heads))

    token_to_explain = attention_len - len(generated_text) + int(token_to_explain.split('_')[0])
    logger.info(f'Using token index {token_to_explain} for explaining')

    # ---***------***------***------***------***------***------***------***------***------***------***------***---
    # ---***--- Learn causal Structure ---***---

    time_struct = []  # list of runtime results for learning the structure for different heads
    time_reason = []  # list of runtime results for recovering an explanation for different heads

    expla_list_all = [None] * num_heads
    explainer_all = [None] * num_heads
    timing_all = [None] * num_heads
    head_importance = [0] * num_heads

    # state_causal_explainers[0] = token_to_explain
    # state_causal_explainers[1] = []
    state_causal_explainers = [token_to_explain]
    # state_causal_explainers.append(dict())

    total_weights = [0 for _ in range(576)]  # weights for image tokens (24 x 24 tokens)

    for head_id in heads_to_analyse:  # ToDo: Run in parallel (threading/multiprocessing; a worker for head)
        head_attention = full_attention[head_id]  # alias for readability

        #  ---***------***--- Text causal graph ---***------***---
        text_expla, text_expl, timing = tokens_analysis(head_attention, list(range(first_im_token_idx+576, token_to_explain+1)),
                                                        token_of_interest=token_to_explain,
                                                        number_of_samples=degrees_of_freedom, p_val_thrshold=threshold, max_search_range=search_range, 
                                                        verbose=False)
        txt_node_labels = dict()
        for v in text_expla:
            # print(f'attention len: {attention_len}  -  Generated len: {len(generated_text)}  +  node: {v}, idx={attention_len - len(generated_text) + v}')
            idx = v - (attention_len - len(generated_text))
            if idx >= 0:
                txt_node_labels[v] = generated_text[idx]
        #  End: *------***--- Text causal graph ---***------***---
        

        w = head_attention[token_to_explain, :]
        w_img = w[first_im_token_idx:(first_im_token_idx+576)]
        # im_entropy = -np.nansum(w_img * np.log(w_img))
        # total_entropy = -np.nansum(w * np.log(w))
        
        # print(f'{head_id}: total_entropy: {total_entropy}, image entropy: {im_entropy}, entropy diff: im - total: {im_entropy - total_entropy}')
        num_high_att = max(2, sum(w > att_th))

        num_image_tokens = min(num_high_att, max_num_image_tokens)  # number of image tokens to select for analysis

        relevant_image_idx = get_relevant_image_tokens(class_token=token_to_explain, 
                                                    attention_matrix=head_attention,
                                                    first_token=first_im_token_idx, 
                                                    num_top_k_tokens=num_image_tokens)

        relevant_gen_idx = get_relevant_text_tokens(class_token=token_to_explain, attention_matrix=head_attention, att_th=att_th, first_image_token=first_im_token_idx)
        relevant_tokens = relevant_image_idx + relevant_gen_idx + [token_to_explain]

        # print(f'Self: {head_attention[token_to_explain, token_to_explain]}')
        # att_th = head_attention[token_to_explain, token_to_explain]
        # att_th = np.median(w[first_im_token_idx+576:])
        # print(f'Attentnion threshold: {att_th}')
        # relevant_tokens = set(np.where(w >= att_th)[0])
        # relevant_tokens.add(token_to_explain)
        # relevant_tokens = list(relevant_tokens)
        # relevant_tokens = [v for v in relevant_tokens if v >= first_im_token_idx]
        # print('relevant tokens', relevant_tokens)

        expla_list, explainer, timing = tokens_analysis(head_attention, relevant_tokens,
                                                        token_of_interest=token_to_explain,
                                                        number_of_samples=degrees_of_freedom, p_val_thrshold=threshold, max_search_range=search_range, 
                                                        verbose=False)

        expla_list_all[head_id] = expla_list
        explainer_all[head_id] = explainer
        timing_all[head_id] = timing

        # calculate Head Importance
        im_expla_tokens_list = [v for v in expla_list if (v >= first_im_token_idx) and (v < first_im_token_idx + 576)]  # only image explanation
        ci_test = explainer.ci_test
        prev_num_records = ci_test.num_records
        ci_test.num_records = len(im_expla_tokens_list)
        weights_list = []
        for im_expla_tok in im_expla_tokens_list:
            cond_set = tuple(set(im_expla_tokens_list) - {im_expla_tok})
            p_val = min(ci_test.calc_statistic(im_expla_tok, token_to_explain, cond_set), 1)  # avoid inf
            weights_list.append(1-p_val)
        ci_test.num_records = prev_num_records

        # print(f'*** Head: {head_id} -- weights: {weights_list}')
        # if len(weights_list) == 0:
        #     head_importance[head_id] = 0
        # else:
        #     head_importance[head_id] = np.mean(weights_list)
        head_importance[head_id] = max(w_img) / max(w[first_im_token_idx+576:])

        for im_expla_tok, im_expla_weight in zip(im_expla_tokens_list, weights_list):
            total_weights[im_expla_tok-first_im_token_idx] += im_expla_weight

        # if len(im_expla_tokens_list) > 0:
        #     head_importance[head_id] = np.median(w[im_expla_tokens_list])
        # else:
        #     head_importance[head_id] = 0

        # p_vals_dict = calculate_explanation_pvals(explainer, token_to_explain, search_range)
        # p_weights_im_tokens = [
        #     (1-p_vals_dict[v])*w[v] for v in expla_list if (v >= first_im_token_idx) and (v < first_im_token_idx + 576)
        # ]
        # if len(p_weights_im_tokens) == 0:
        #     head_importance[head_id] = 0
        # else:
        #     head_importance[head_id] = np.median(p_weights_im_tokens)

        # if len(expla_list) > 0:
        #     # head_importance[head_id] = np.median(w[expla_list])
        #     head_importance[head_id] = np.median(sorted(w)[-max_num_image_tokens:])
        # else:
        #     head_importance[head_id] = 0
            
        txt = '{head}:    {importance:.2f}    / 100'.format(head=head_id, importance=head_importance[head_id]*100)
        logger.info(f'Head: {head_id}: importance: {txt}')

        
        time_struct.append(timing['structure'])
        time_reason.append(timing['reasoning'])
        im_expla_rel_idx = [v-first_im_token_idx for v in im_expla_tokens_list]  # only image

        # print(f'head {head_id}, importance: {head_importance[head_id]:.3f}, above {att_th}: {num_high_att}')

        # plot results
        logger.info('Max: *******', max(total_weights))
        if max(total_weights) > 0:
            norm_total_weights =  [v/max(total_weights) for v in total_weights]
        else:
            norm_total_weights = total_weights
        im_t = recovered_image.copy()
        im_heat_total = show_tokens_on_image(list(range(576)), im_t, norm_total_weights)
        im_heat_edit_t = ImageDraw.Draw(im_heat_total)
        im_heat_edit_t.text((10, 10), txt, fill=(255, 255, 255))
        im_heat_total = im_heat_edit_t._image

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(range(num_heads), head_importance)
        ax.grid(True)
        xmin, xmax, ymin, ymax = ax.axis()
        ax.axis([1, 32, -ymax*0.01, ymax])
        # ax.set_position([0, 0, 1, 1])
        fig.canvas.draw()
        im_head_importance = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        plt.close()

        # attentnion values
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        h =  [max(w[first_im_token_idx:576+first_im_token_idx])] + list(w[first_im_token_idx+576:])
        ax.bar(range(len(h)), h)
        ax.grid(True)
        fig.canvas.draw()
        im_att_bar = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        plt.close()

        im_heat = create_im_tokens_marks(recovered_image, im_expla_rel_idx, txt=txt)
        # im_1 = recovered_image.copy()
        # # im_heat = show_tokens_on_image(im_expla_rel_idx, im_1, weights_list)
        # im_heat = show_tokens_on_image(im_expla_rel_idx, im_1)
        # im_heat_edit = ImageDraw.Draw(im_heat)
        # im_heat_edit.text((10, 10), txt, fill=(255, 255, 255))
        # im_heat = im_heat_edit._image

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(head_importance, '.-')
        ax.grid(True)
        fig.canvas.draw()
        im_pl = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        plt.close()

        nodes_set = set(expla_list)
        nodes_set.add(token_to_explain)
        subgraph = copy_sub_graph(explainer.graph, nodes_set)
        fig = draw_graph(subgraph, show=False)
        fig.canvas.draw()
        # im_graph = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        plt.close()

        # nodes_set = set(text_expla)
        # nodes_set.add(token_to_explain)
        # subgraph = copy_sub_graph(text_expl.graph, nodes_set)
        # fig = draw_graph(subgraph, show=False, node_labels=node_labels)
        # fig.canvas.draw()
        im_graph = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        # plt.close()

        node_labels = dict()
        for tok in expla_list:
            if tok in txt_node_labels:  # if token is text
                node_labels[tok] = txt_node_labels[tok]
                continue
            im_idx = tok - first_im_token_idx
            if im_idx < 0 or im_idx >= 576:  # if token is not image
                continue
            im_tok = crop_token(recovered_image, im_idx, pad=2)
            node_labels[tok] = im_tok.resize((45, 45))

        idx = token_to_explain - (attention_len - len(generated_text))
        node_labels[token_to_explain] = generated_text[idx]
        
        nodes_set = set(expla_list)
        nodes_set.add(token_to_explain)
        fig = draw_pds_tree(explainer.results[token_to_explain]['pds_tree'], explainer.graph, node_labels=node_labels,
                          node_size_factor=1.4)
        if fig is None:
            fig = plt.figure()
        fig.canvas.draw()
        im_graph = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
        plt.close()

        gallery_image_list.append(im_heat)
        gallery_graph_list.append(im_graph)
        gallery_bar_graphs.append(im_att_bar)
        # gallery_image_list.append(im_pl)
    
    state_causal_explainers.append(explainer_all)  # idx 1
    state_causal_explainers.append(expla_list_all) # idx 2
    return gallery_image_list + gallery_graph_list + gallery_bar_graphs, state_causal_explainers #im_heat_total #im_head_importance

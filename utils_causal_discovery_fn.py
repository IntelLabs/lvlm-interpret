import time
import numpy as np
from itertools import combinations

from PIL import Image, ImageEnhance
import torch

import matplotlib.pyplot as plt

try:
    from causal_discovery_algs import LearnStructOrderedICD
except ImportError:
    print("Warning: causal discovery pending update.")
    LearnStructOrderedICD = None

from graphical_models import PAG
from causal_reasoning import CLEANN


def get_expla_set_per_rad(pds_tree):
    root_node = pds_tree.origin
    expla_lists_per_rad = {0: root_node}
    children = pds_tree.children
    rad = 1
    while len(children) > 0:
        expla_lists_per_rad[rad] = set()  # initialize an explanation set at range rad
        children_of_children = []
        for child in children:
            expla_lists_per_rad[rad].add(child.origin)
            children_of_children += child.children
        rad += 1
        children = children_of_children
    return expla_lists_per_rad



def get_relevant_image_tokens(class_token, attention_matrix, first_token, num_top_k_tokens) -> list:
    """
    Find the indexes of the image tokens for which the class tokens most attend (highest attention)

    :param class_token:
    :param attention_matrix:
    :param first_token:
    :param num_top_k_tokens:
    """
    weights = attention_matrix[class_token, first_token:(first_token+576)]
    sorting_indexes = np.argsort(weights)[::-1]  # descending sorting indexes
    all_indexes = list(range(576))
    top_k_idx = [all_indexes[sorting_indexes[i]] for i in range(num_top_k_tokens)]
    top_k_idx = [t + first_token for t in top_k_idx]  # add index offset
    return top_k_idx


def get_relevant_prompt_tokens(class_token, attention_matrix, att_th, first_image_token) -> list:
    weights = attention_matrix[class_token, :first_image_token]
    relevent_prompt_tokens = np.where(weights > att_th)[0]
    return list(relevent_prompt_tokens)


def get_relevant_text_tokens(class_token, attention_matrix, att_th, first_image_token) -> list:
    """
    Get the indexes of the text tokens after the image (not including the prompt)
    for which the class tokens highly attends (attention above the threshold)
    """
    weights = attention_matrix[class_token, (first_image_token+576):class_token]
    idxs = np.where(weights > att_th)[0]
    relevent_gen_tokens = [t + (first_image_token+576) for t in idxs]
    return relevent_gen_tokens


def tokens_analysis(attention_matrix, tokens_idx_list, token_of_interest,
                  number_of_samples, p_val_thrshold, max_search_range=None, verbose=True):
    explanation_list, cleann_explainer, runtimes = create_explanation(attention_matrix, tokens_idx_list, token_of_interest,
                                                                      number_of_samples, p_val_thrshold, max_search_range, 
                                                                      verbose=verbose)
    explanation_list = sorted(explanation_list)
    if verbose:
        print(f'len {len(explanation_list)}', explanation_list)
    return explanation_list, cleann_explainer, runtimes


def create_explanation(attention_matrix, tokens_idx_list, token_of_interest,
                       number_of_samples, p_val_thrshold, max_search_range=None, verbose=True):
    cleann_explainer = CLEANN(
        attention_matrix=attention_matrix,
        num_samples=number_of_samples,
        p_val_th=p_val_thrshold,
        explanation_tester=None,
        nodes_set=set(tokens_idx_list),

    )
    cond_indep_test = cleann_explainer.ci_test
    structure_learner = LearnStructOrderedICD(set(tokens_idx_list), sorted(tokens_idx_list), cond_indep_test,
                                              is_selection_bias=False)

    runtimes = {'structure': None, 'reasoning': None}
    t0 = time.time()
    structure_learner.learn_structure_global()
    t1 = time.time()
    runtimes['structure'] = t1-t0
    if verbose:
        print(f'Structure learning time {t1 - t0} seconds.')

    cleann_explainer.graph = structure_learner.graph
    t0 = time.time()
    explanation = cleann_explainer.explain(token_of_interest, max_range=max_search_range)
    t1 = time.time()
    runtimes['reasoning'] = t1-t0
    if verbose:
        print(f'Explanation deduction time {t1 - t0} seconds.')
    explanation_list = [v for v in explanation[0][0]]
    return explanation_list, cleann_explainer, runtimes


def copy_sub_graph(full_graph: PAG, nodes_of_interest: set) -> PAG:
    sub_graph = PAG(nodes_of_interest)
    sub_graph.create_empty_graph()
    for node_i, node_j in combinations(nodes_of_interest, 2):
        if full_graph.is_connected(node_i, node_j):
            edge_at_i = full_graph.get_edge_mark(node_j, node_i)
            edge_at_j = full_graph.get_edge_mark(node_i, node_j)
            sub_graph.add_edge(node_i, node_j, edge_at_i, edge_at_j)
    return sub_graph

# def create_preprocessed_image(in_image):
#     img_std = torch.tensor(image_processor.image_std).view(3,1,1)
#     img_mean = torch.tensor(image_processor.image_mean).view(3,1,1)
#     img_recover = in_image * img_std + img_mean
#     return to_pil_image(img_recover)


def show_tokens_on_image(selected_image_tokens, pil_image, weights=None):
    if weights is None or len(weights)==0:
        weights_n = [0.7] * len(selected_image_tokens)
    else:
        mx = 1  # max(weights)
        weights_n = [v/mx for v in weights]

    tokens_mask = np.zeros(576)
    for i, tok in enumerate(selected_image_tokens):
        tokens_mask[tok] = weights_n[i]
    cmap = plt.get_cmap('jet')
    im_mask = tokens_mask.reshape((24, 24))
    im_mask = cmap(im_mask)
    a_im = Image.fromarray((im_mask[:, :, :3] * 255).astype(np.uint8)).resize((336, 336), Image.BICUBIC)
    a_im.putalpha(128)
    new_im = pil_image.copy()
    new_im.paste(a_im, mask=a_im)
    return new_im


def calculate_explanation_pvals(explainer_instance, target_node, max_range=None):
    if target_node not in explainer_instance.results:
        raise "explainer should have initially been run."
    if max_range is None:
        max_range = explainer_instance.results[target_node]['max_pds_tree_depth']

    ci_test = explainer_instance.ci_test  # alias
    pvals = dict()

    cond_set = ()  # initial conditioning set
    prev_res_set = set()
    for r in range(1, max_range):
        res_set = explainer_instance.explain(target_node, max_range=r)[0][0]
        for v in res_set.difference(prev_res_set):
            pvals[v] = min(ci_test.calc_statistic(v, target_node, cond_set), 1)
        cond_set = tuple(res_set)
        prev_res_set = res_set
    return pvals


def image_token_to_xy(image_token, n_x_tokens=24, n_y_tokens=24, token_width=14, token_height=14):

    x_pos = (image_token % n_x_tokens) * token_width
    y_pos = (image_token // n_y_tokens) * token_height
    return x_pos, y_pos


def crop_token(in_im, image_token, n_x_tokens=24, n_y_tokens=24, pad=None):
    im_width, im_height = in_im.size
    token_width = im_width // n_x_tokens
    token_height = im_height // n_y_tokens
    x_pos, y_pos = image_token_to_xy(image_token, n_x_tokens, n_y_tokens, token_width, token_height)
    left= x_pos
    right = left + token_width - 1
    top = y_pos
    bottom = top + token_height - 1
    im_token = in_im.crop((left, top, right, bottom))
    if pad is None:
        return im_token
    else:
        left_pad = max(0, left-pad*token_width)
        right_pad = min(im_width-1, right+pad*token_width)
        top_pad = max(0, top-pad*token_height)
        bottom_pad = min(im_height-1, bottom+pad*token_height)
        # print(left_pad, right_pad, top_pad, bottom_pad)
        enhancer = ImageEnhance.Brightness(in_im)
        new_im = enhancer.enhance(0.5)
        pad_image = new_im.crop((left_pad, top_pad, right_pad, bottom_pad))
        pad_image.paste(im_token, (left-left_pad, top-top_pad))
        return pad_image

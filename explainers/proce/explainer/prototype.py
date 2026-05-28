#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:06:31 2020

@author: trduong
"""

import torch


def find_proto(original_latent, pos_latent, neg_latent, k_instance):
    ori_pos_dist = torch.cdist(original_latent, pos_latent, p=1)
    ori_neg_dist = torch.cdist(original_latent, neg_latent, p=1)

    _, pos_index = torch.topk(-ori_pos_dist, k=k_instance)
    _, neg_index = torch.topk(-ori_neg_dist, k=k_instance)

    pos_proto = torch.mean(pos_latent[pos_index], dim=1)
    neg_proto = torch.mean(neg_latent[neg_index], dim=1)

    return pos_proto, neg_proto


def get_pos_neg_latent(prediction, z_representation):
    pos_locs = torch.eq(prediction, 1.0)
    neg_locs = torch.eq(prediction, 0.0)

    pos_z = z_representation[pos_locs]
    neg_z = z_representation[neg_locs]
    return pos_z, neg_z

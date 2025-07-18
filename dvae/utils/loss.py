#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt
"""
import torch

def loss_ISD(x, y):
    y = y + 1e-10
    ret = torch.sum( x/y - torch.log(x/y) - 1)
    return ret

def loss_KLD(z_mean, z_logvar, z_mean_p=0, z_logvar_p=0):
    ret = -0.5 * torch.sum(z_logvar - z_logvar_p 
                - torch.div(z_logvar.exp() + (z_mean - z_mean_p).pow(2), z_logvar_p.exp()+1e-10))
    return ret

def loss_JointNorm(x, y, nfeats=3):
    seq_len, bs, _ = x.shape
    x = x.reshape(seq_len, bs, -1, nfeats)
    y = y.reshape(seq_len, bs, -1, nfeats)
    ret = torch.sum(torch.norm(x-y, dim=-1))
    return ret

def loss_MPJPE(x, y, nfeats=3):
    seq_len, bs, _ = x.shape
    x = x.reshape(seq_len, bs, -1, nfeats)
    y = y.reshape(seq_len, bs, -1, nfeats)
    ret = (x-y).norm(dim=-1).mean(dim=-1).sum()
    return ret

def loss_rec_plus_KLD(x_pred, x_true, z_mean, z_logvar, z_mean_p, z_logvar_p, beta=1.0):
    # Reconstruction loss: MSE over the sequence
    recon_loss = torch.nn.functional.mse_loss(x_pred, x_true, reduction='mean')

    # KL divergence between q(z|x,h) ~ N(z_mean, exp(z_logvar)) (approximate posterior)
    # and p(z|h) ~ N(z_mean_p, exp(z_logvar_p)) (prior)
    kl_loss = -0.5 * torch.sum(
        z_logvar - z_logvar_p
        - ((z_logvar.exp() + (z_mean - z_mean_p).pow(2)) / (z_logvar_p.exp() + 1e-10))
    )
    kl_loss = kl_loss / x_true.shape[1]  # normalize by batch_size

    return recon_loss + beta * kl_loss



# def loss_ISD(x, y):
#     seq_len, bs, _ = x.shape
#     ret = torch.sum( x/y - torch.log(x/y) - 1)
#     ret = ret / (bs * seq_len)
#     return ret

# def loss_KLD(z_mean, z_logvar, z_mean_p=0, z_logvar_p=0):
#     if len(z_mean.shape) == 3:
#         seq_len, bs, _ = z_mean.shape
#     elif len(z_mean.shape) == 2:
#         seq_len = 1
#         bs, _ = z_mean.shape
#     ret = -0.5 * torch.sum(z_logvar - z_logvar_p 
#                 - torch.div(z_logvar.exp() + (z_mean - z_mean_p).pow(2), z_logvar_p.exp()))
#     ret = ret / (bs * seq_len)
#     return ret

# def loss_JointNorm(x, y, nfeats=3):
#     seq_len, bs, _ = x.shape
#     x = x.reshape(seq_len, bs, -1, nfeats)
#     y = y.reshape(seq_len, bs, -1, nfeats)
#     return torch.mean(torch.norm(x-y, dim=-1))




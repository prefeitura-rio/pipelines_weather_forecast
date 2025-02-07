# -*- coding: utf-8 -*-
import torch


def make_grid(input, device):
    B, C, H, W = input.size()
    # mesh grid
    xx = torch.arange(0, W, device=device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    return grid


def warp(input, flow, grid, mode="bilinear", padding_mode="zeros"):
    B, C, H, W = input.size()
    vgrid = grid + flow

    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(
        input, vgrid, padding_mode=padding_mode, mode=mode, align_corners=True
    )
    return output

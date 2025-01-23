# -*- coding: utf-8 -*-
from functools import partial

import numpy as np
import torch


def MSE(pred, truth):
    pred[np.logical_or(pred <= 0, ~np.isfinite(pred))] = 0
    truth[np.logical_or(truth <= 0, ~np.isfinite(truth))] = 0
    return np.mean((pred - truth) ** 2)


def logMSE(pred, truth):
    pred[np.logical_or(pred <= 0, ~np.isfinite(pred))] = 0
    truth[np.logical_or(truth <= 0, ~np.isfinite(truth))] = 0
    return np.mean((np.log1p(np.nan_to_num(pred)) - np.log1p(np.nan_to_num(truth))) ** 2)


def MAE(pred, truth):
    pred[np.logical_or(pred <= 0, ~np.isfinite(pred))] = 0
    truth[np.logical_or(truth <= 0, ~np.isfinite(truth))] = 0
    return np.mean(np.abs(pred - truth))


def logMAE(pred, truth):
    pred[np.logical_or(pred <= 0, ~np.isfinite(pred))] = 0
    truth[np.logical_or(truth <= 0, ~np.isfinite(truth))] = 0
    return np.mean(np.abs(np.log1p(np.nan_to_num(pred)) - np.log1p(np.nan_to_num(truth))))


# [0,1]
def CSI(pred, truth, threshold=2):
    binary_pred = np.nan_to_num(pred) > threshold
    binary_truth = np.nan_to_num(truth) > threshold
    return np.sum(np.logical_and(binary_pred, binary_truth)) / (
        np.sum(binary_pred)
        + np.sum(binary_truth)
        - np.sum(np.logical_and(binary_pred, binary_truth))
    )


def csi_denominator(pred, truth, threshold=2):
    binary_pred = np.nan_to_num(pred) > threshold
    binary_truth = np.nan_to_num(truth) > threshold
    TP = np.sum(np.logical_and(binary_pred, binary_truth))
    FN = np.sum(np.logical_and(np.logical_not(binary_pred), binary_truth))
    FP = np.sum(np.logical_and(binary_pred, np.logical_not(binary_truth)))
    return TP + FN + FP


def true_positive(pred, truth, threshold=2):
    binary_pred = np.nan_to_num(pred) > threshold
    binary_truth = np.nan_to_num(truth) > threshold
    TP = np.sum(np.logical_and(binary_pred, binary_truth))
    return TP


def csi_denominator_torch(pred, truth, threshold=2, dim=None):
    binary_pred = torch.nan_to_num(pred) > threshold
    binary_truth = torch.nan_to_num(truth) > threshold
    TP = torch.sum(torch.logical_and(binary_pred, binary_truth), dim=dim)
    FN = torch.sum(torch.logical_and(torch.logical_not(binary_pred), binary_truth), dim=dim)
    FP = torch.sum(torch.logical_and(binary_pred, torch.logical_not(binary_truth)), dim=dim)
    return TP + FN + FP


def true_positive_torch(pred, truth, threshold=2, dim=None):
    binary_pred = torch.nan_to_num(pred) > threshold
    binary_truth = torch.nan_to_num(truth) > threshold
    TP = torch.sum(torch.logical_and(binary_pred, binary_truth), dim=dim)
    return TP


def CSI_total(TP, FN, FP):
    return TP / (TP + FN + FP)


def maxpool(image):
    conv = torch.nn.MaxPool2d(kernel_size=5, stride=2)
    array_conv = conv(torch.tensor(image)[None, None, :, :])
    return array_conv


def csi_neighborhood_denominator(pred, truth, threshold=2):
    pred_conv = maxpool(pred)
    truth_conv = maxpool(truth)
    return csi_denominator(pred_conv, truth_conv, threshold)


def true_positive_neighborhood(pred, truth, threshold=2):
    pred_conv = maxpool(pred)
    truth_conv = maxpool(truth)
    return true_positive(pred_conv, truth_conv, threshold)


# [-1/3,1]
def ETS(pred, truth, threshold=2):
    binary_pred = np.nan_to_num(pred) > threshold
    binary_truth = np.nan_to_num(truth) > threshold
    R = np.sum(binary_pred) * np.sum(binary_truth) / len(pred.flatten())
    return (np.sum(np.logical_and(binary_pred, binary_truth)) - R) / (
        np.sum(binary_pred)
        + np.sum(binary_truth)
        - np.sum(np.logical_and(binary_pred, binary_truth))
        - R
    )


metrics_dict = {
    "MSE": MSE,
    "MAE": MAE,
    "log-MSE": logMSE,
    "log-MAE": logMAE,
    "CSI1": partial(CSI, threshold=1),
    "ETS1": partial(ETS, threshold=1),
    "CSI2": partial(CSI, threshold=2),
    "ETS2": partial(ETS, threshold=2),
    "CSI4": partial(CSI, threshold=4),
    "ETS4": partial(ETS, threshold=4),
    "CSI8": partial(CSI, threshold=8),
    "ETS8": partial(ETS, threshold=8),
    "CSI16": partial(CSI, threshold=16),
}

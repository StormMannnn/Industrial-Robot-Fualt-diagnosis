import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    mape = np.abs((pred - true) / true)
    mape = np.where(mape > 5, 0, mape)
    return np.mean(mape)


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def metric_classification(preds, trues):
    """
    计算分类任务的评估指标
    """
    # 修复：preds和trues已经是numpy数组，不需要torch.argmax
    accuracy = (preds == trues).mean()

    # 计算其他指标
    precision = precision_score(trues, preds, average='weighted', zero_division=0)
    recall = recall_score(trues, preds, average='weighted', zero_division=0)
    f1 = f1_score(trues, preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(trues, preds)

    return precision, recall, f1, conf_matrix
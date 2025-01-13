import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import torch.nn as nn


def MSE(preds, true):
    squaredError = []
    for i in range(len(preds)):
        dis = true[i] - preds[i]
        squaredError.append(dis * dis)
    return sum(squaredError) / len(squaredError)


def MAE(preds, true):
    absError = []
    for i in range(len(preds)):
        dis = true[i] - preds[i]
        absError.append(abs(dis))
    return sum(absError) / len(absError)


def RMSE(preds, true):
    squaredError = []
    absError = []
    for i in range(len(preds)):
        dis = true[i] - preds[i]
        squaredError.append(dis * dis)
        absError.append(abs(dis))
    from math import sqrt
    return sqrt(sum(squaredError) / len(squaredError))


def AUC(true, preds):
    return roc_auc_score(true, preds)


def eval_4_adao2b(model, val_data, batch_size, device):
    val_dataloader = DataLoader(val_data, batch_size, shuffle=False)
    model.eval()
    pred_ans = []
    label = []
    total_epoch_loss = 0
    Loss = nn.BCELoss() 
    for step, (can, base_score, act, rew) in enumerate(val_dataloader):
        can = can.to(device)
        base_score = base_score.to(device)
        act = act.to(device)
        rew = rew.to(device)
        preds = model.evaluate(can, base_score, act)
        if step < len(val_dataloader) - 1: # last step may be less than batchsize, don't count in loss when evaluating
            total_epoch_loss += Loss(preds, rew.reshape(-1)).item()
        pred_ans.append(preds.detach().cpu().numpy())
        label.append(rew.reshape(-1).detach().cpu().numpy())

    pred = np.concatenate(pred_ans).astype("float64")
    true = np.concatenate(label).astype("float64")
    auc = AUC(true, pred)
    mse = MSE(pred, true)

    return auc, mse, total_epoch_loss/(len(val_dataloader)-1)


def total_average_reward(user_feedback):
    user_feedback = user_feedback.reshape(-1)
    total_avg_rw = user_feedback.sum() / user_feedback.shape[0]

    return total_avg_rw

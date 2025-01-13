import argparse
import copy
import os
import random
from time import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from Model.AdaO2B import AdaO2B
from Model.EXP3_B import EXP3_B
from Model.SBUCB import SBUCB
from Model.BLTS_B import BLTS_B
from utils.data import (Data_4_AdaO2B, get_candidata_vector, load_base_model,
                        prep_data_4_adao2b, reservoir_sampling,
                        data_dependent_sampling)
from utils.evaluation import eval_4_adao2b


class DefaultConfig(object):
    def __init__(self, base_model, base_model_save_name, K, history_data_name):
        self.model = 'AdaO2B'

        self.base_model = base_model
        self.base_model_save_name = base_model_save_name
        self.N = 40
        self.B = 5000

        self.K = K
        self.base_model_batch_list = [
            i for i in range(self.N - self.K, self.N)
        ]
        self.use_all_history_data = 0
        self.history_result_path = "../Result/" + self.base_model + "/"
        self.history_avg_reward_file_name = self.base_model_save_name + "_online_data1"

        self.data_selection = 'S'  # ['S','R','D']

        self.adao2b_path = '../Data/AdaO2B/'
        self.adao2b_ckpt_path = '../CKPT/AdaO2B/'
        self.adao2b_ckpt_name = ''
        self.history_data_path = self.adao2b_path + self.base_model + " _" + history_data_name + "_online_data1_adao2b.csv"

        self.feature_dimension = 10

        self.metric = 'auc'
        self.verbose = 10

        self.device = 'cuda:1'

        self.max_epoch = 300
        self.batch_size = 1024
        self.lr = 0.001
        self.weight_decay = 1e-6

        self.res_name = ""

        self.seed_num = 2022


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(opt):
    if opt.base_model == "SBUCB":
        base_model = SBUCB()
    elif opt.base_model == "BLTS_B":
        base_model = BLTS_B()
    elif opt.base_model == "EXP3_B":
        base_model = EXP3_B()

    base_model_list = load_base_model(base_model, opt.base_model_save_name,
                                      opt.base_model_batch_list)

    data_4_adao2b = prep_data_4_adao2b(base_model_list, opt.history_data_path,
                                       opt.base_model_batch_list,
                                       opt.use_all_history_data)

    train_data, val_data = train_test_split(data_4_adao2b,
                                            train_size=0.8,
                                            random_state=opt.seed_num)

    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True)

    o2b_model = AdaO2B(opt.K, opt.feature_dimension,
                       opt.base_model == "EXP3_B")
    o2b_model.to(opt.device)
    optimizer = o2b_model.get_optimizer(opt.lr, opt.weight_decay)

    best_auc = 0
    best_mse = 10000
    best_iter = 0

    for epoch in range(opt.max_epoch):
        t1 = time()
        o2b_model.train()
        total_epoch_loss = 0
        for can, base_score, act, rew in train_dataloader:
            can = can.to(opt.device)
            base_score = base_score.to(opt.device)
            act = act.to(opt.device)
            rew = rew.to(opt.device)

            loss = o2b_model.calculate_loss(can, base_score, act, rew)
            total_epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        t2 = time()

        auc, mse, val_logloss = eval_4_adao2b(o2b_model, val_data,
                                              opt.batch_size, opt.device)

        if auc > best_auc and epoch > 0:
            best_auc, best_mse, best_iter = auc, mse, epoch
            torch.save(
                o2b_model.state_dict(), opt.adao2b_ckpt_path +
                opt.adao2b_ckpt_name + "_best_adao2b.pth")

        if epoch % opt.verbose == 0:
            print('Epoch %d [%.1f s]:' % (epoch, t2 - t1))
            print('Train Loss = ', total_epoch_loss / len(train_dataloader))
            print('Val AUC = %.4f, MSE = %.4f, Logloss = %4f[%.1f s]' %
                  (auc, mse, val_logloss, time() - t2))
            print("------------------------------------------")

    print("train end\nBest Epoch%d:  auc = %.4f. " % (best_iter, best_auc))

    best_model = AdaO2B(opt.K, opt.feature_dimension,
                        opt.base_model == "EXP3_B")
    best_model.to(opt.device)
    best_model.load_state_dict(
        torch.load(opt.adao2b_ckpt_path + opt.adao2b_ckpt_name +
                   "_best_adao2b.pth"))

    print("\n=============best model=============")
    auc, mse, logloss = eval_4_adao2b(best_model, train_data, opt.batch_size,
                                      opt.device)
    print('Train AUC = %.4f, MSE = %.4f, Logloss = %4f' % (auc, mse, logloss))
    print("------------------------------------")
    val_auc, val_mse, val_logloss = eval_4_adao2b(best_model, val_data,
                                                  opt.batch_size, opt.device)
    print('Val AUC = %.4f, MSE = %.4f, Logloss = %4f' %
          (val_auc, mse, val_logloss))
    print("==============================================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('--epoch', type=int, default=100, help='epoch')
    parser.add_argument('--base_model',
                        default='SBUCB',
                        choices=["SBUCB", "EXP3_B", "BLTS_B"])
    parser.add_argument('--K',
                        default=10,
                        type=int,
                        help='number of base model and data buffers')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1024,
                        help='batch_size')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='learning rate')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-6,
                        help='weight decay')
    parser.add_argument('--use_all_history_data',
                        type=int,
                        default=0,
                        choices=[0, 1])
    parser.add_argument('--adao2b_ckpt_name', help="adao2b_ckpt_name")
    parser.add_argument('--base_model_save_name', help="base_model_save_name")
    parser.add_argument('--history_data_name', help="history_data_name")
    parser.add_argument('--data_selection',
                        help="data_selection",
                        choices=['S', 'R', 'D'],
                        default='S')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--verbose', type=int, default=10)

    args = parser.parse_args()
    opt = DefaultConfig(args.base_model, args.base_model_save_name, args.K,
                        args.history_data_name)
    # opt.max_epoch = args.epoch
    opt.batch_size = args.batch_size
    opt.lr = args.lr
    opt.weight_decay = args.weight_decay
    opt.use_all_history_data = args.use_all_history_data
    opt.adao2b_ckpt_name = args.adao2b_ckpt_name
    opt.data_selection = args.data_selection
    opt.seed_num = args.seed

    setup_seed(opt.seed_num)

    if opt.data_selection == 'S':
        opt.base_model_batch_list = [i for i in range(opt.N - opt.K, opt.N)]
    elif opt.data_selection == 'R':
        opt.base_model_batch_list = reservoir_sampling(opt.N, opt.K)
        print("base_model_batch_list:", opt.base_model_batch_list)
    elif opt.data_selection == 'D':
        opt.base_model_batch_list = data_dependent_sampling(
            opt.history_result_path, opt.history_avg_reward_file_name, opt.B,
            opt.K)

    print('\n'.join(['%s:%s' % item for item in opt.__dict__.items()]))

    train(opt)
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import argparse
import random
import copy
import os
import time
from scipy.special import expit
from tqdm import tqdm
from Model.AdaO2B import AdaO2B
from Model.SBUCB import SBUCB
from Model.EXP3_B import EXP3_B
from Model.BLTS_B import BLTS_B
from utils.data import (get_candidata_vector, load_base_model,
                        reservoir_sampling, data_dependent_sampling)
from utils.simulator import get_batch_feedback
from utils.evaluation import total_average_reward
from itertools import product


class DefaultConfig(object):
    def __init__(self, rec_model, base_model_save_name, K, data_index=2):
        self.rec_model = rec_model
        self.device = "cuda:1"

        self.data_index = data_index
        self.feature_dimension = 10
        self.N = 40
        self.B = 5000
        self.candidate_size = 10

        # simulator
        self.w = np.random.normal(0.1, 0.01,
                                  (self.feature_dimension, 1))  # reward weight

        self.base_model_save_name = base_model_save_name
        self.K = K
        self.base_model_batch_list = [
            i for i in range(self.N - self.K, self.N)
        ]
        self.adao2b_path = '../Data/AdaO2B/'
        self.adao2b_ckpt_path = '../CKPT/AdaO2B/'
        self.adao2b_ckpt_name = ''
        self.history_result_path = "../Result/" + self.rec_model + "/"
        self.history_avg_reward_file_name = self.base_model_save_name + "_online_data1"

        # result
        self.result_path = "../Result/"
        self.res_name = ""

        # model save and load
        self.rec_model_load_name = ""
        self.rec_model_save_name = ""

        # SBUCB
        self.SBUCB_mu = 0.4

        # BLTS_B
        self.BLTS_B_mu = 0.8

        # EXP3_B
        self.EXP3_B_delta = 0.1

        self.seed_num = 2022


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def bacth_test():
    if opt.rec_model == "SBUCB":
        rec_model = SBUCB(mu=opt.SBUCB_mu, feature_d=opt.feature_dimension)
    elif opt.rec_model == "BLTS_B":
        rec_model = BLTS_B(mu=opt.BLTS_B_mu, feature_d=opt.feature_dimension)
    elif opt.rec_model == "EXP3_B":
        rec_model = EXP3_B(N=opt.N,
                           B=opt.B,
                           C=opt.candidate_size,
                           delta=opt.EXP3_B_delta,
                           feature_d=opt.feature_dimension)

    print("****Load model from: ", opt.rec_model_load_name)
    rec_model.load_model(opt.rec_model_load_name)

    # save result for stastics
    full_user_feedback = []

    t_recommend = 0
    recommend_time = []

    t1 = time.time()
    for n in tqdm(range(opt.N)):
        # print("theta", rec_model.theta)

        rec_state_n = []
        user_feedback_n = []
        rewards_n = []

        for b in range(opt.B):
            # Observe the set of candidate items S_𝑛𝑏
            candidate_state_vector_nb = get_candidata_vector(
                opt.data_index, opt.candidate_size, opt.feature_dimension)

            # Recommend item 𝒔 ∈ S_𝑛𝑏 to the user
            t_recommend_start = time.time()
            rec_state_nb_index = rec_model.bt_recommend(
                candidate_state_vector_nb)
            t_recommend_end = time.time()
            t_recommend += t_recommend_end - t_recommend_start
            recommend_time.append(t_recommend_end - t_recommend_start)

            rec_state_nb = candidate_state_vector_nb[rec_state_nb_index]
            rec_state_n.append(rec_state_nb)

        # Receive batch user feedback
        user_feedback_n = get_batch_feedback(
            np.array(rec_state_n).reshape(opt.B, -1), opt.w)
        rewards_n = copy.deepcopy(user_feedback_n)

        # save batch result
        full_user_feedback.append(user_feedback_n)

    t2 = time.time()
    print("===============Batch Test===============")
    print("rec model: ", opt.rec_model)
    print("dataset index: ", opt.data_index)
    print("total time cost: %fs" % (t2 - t1))
    print("total recommenda time cost: %fs" % (t_recommend))

    np.savetxt(opt.result_path + opt.rec_model + "/" + opt.res_name +
               "_batch_data" + str(opt.data_index) + "_full_user_feedback.txt",
               np.array(full_user_feedback),
               fmt="%d")
    # np.savetxt(
    #     opt.result_path + opt.rec_model + "/" + opt.res_name +
    #     "_batch_data" + str(opt.data_index) + "_recommend_time.txt",
    #     np.array(recommend_time))
    print("*****************total average reward: %4f" %
          (total_average_reward(np.array(full_user_feedback))))
    print("***********************end************************")


def ada_bacth_test():
    setup_seed(opt.seed_num)

    if opt.rec_model == "SBUCB":
        base_model = SBUCB(mu=opt.SBUCB_mu, feature_d=opt.feature_dimension)
    elif opt.rec_model == "BLTS_B":
        base_model = BLTS_B(mu=opt.BLTS_B_mu, feature_d=opt.feature_dimension)
    elif opt.rec_model == "EXP3_B":
        base_model = EXP3_B(N=opt.N,
                            B=opt.B,
                            C=opt.candidate_size,
                            delta=opt.EXP3_B_delta,
                            feature_d=opt.feature_dimension)

    base_model_list = load_base_model(base_model, opt.base_model_save_name,
                                      opt.base_model_batch_list)

    with torch.no_grad():
        adao2b_model = AdaO2B(opt.K, opt.feature_dimension, opt.device)
        adao2b_model.load_state_dict(
            torch.load(opt.adao2b_ckpt_path + opt.adao2b_ckpt_name +
                       "_best_adao2b.pth"))
        adao2b_model.to(opt.device)

        # save result for stastics
        full_user_feedback = []

        t_recommend = 0
        recommend_time = []

        t1 = time.time()
        for n in tqdm(range(opt.N)):
            # print("theta", rec_model.theta)

            rec_state_n = []
            user_feedback_n = []
            rewards_n = []

            for b in range(opt.B):
                # Observe the set of candidate items S_𝑛𝑏
                candidate_state_vector_nb = get_candidata_vector(
                    opt.data_index, opt.candidate_size, opt.feature_dimension)

                # Recommend item 𝒔 ∈ S_𝑛𝑏 to the user
                t_recommend_start = time.time()

                base_rec_score_nb = []
                for tmp_base_model in base_model_list:
                    base_rec_score_nb.append(
                        tmp_base_model.get_rec_score(
                            candidate_state_vector_nb))
                base_rec_score_nb = np.array(base_rec_score_nb).T

                candidate_state_vector_nb_tensor = torch.tensor(
                    candidate_state_vector_nb).type(torch.float32)
                base_rec_score_nb_tensor = torch.tensor(
                    base_rec_score_nb).type(torch.float32)
                # print("=====", candidate_state_vector_nb_tensor.shape,
                #       base_rec_score_nb_tensor.shape)
                # print(candidate_state_vector_nb_tensor,base_rec_score_nb_tensor)
                candidate_state_vector_nb_tensor = candidate_state_vector_nb_tensor.unsqueeze(
                    0).to(opt.device)
                base_rec_score_nb_tensor = base_rec_score_nb_tensor.unsqueeze(
                    0).to(opt.device)

                rec_state_nb_index = adao2b_model.recommend(
                    candidate_state_vector_nb_tensor, base_rec_score_nb_tensor)
                t_recommend_end = time.time()
                t_recommend += t_recommend_end - t_recommend_start
                recommend_time.append(t_recommend_end - t_recommend_start)

                rec_state_nb = candidate_state_vector_nb[rec_state_nb_index]
                # print("---",candidate_state_vector_nb.shape, rec_state_nb_index,rec_state_nb)
                rec_state_n.append(rec_state_nb)

            # Receive batch user feedback
            user_feedback_n = get_batch_feedback(
                np.array(rec_state_n).reshape(opt.B, -1), opt.w)
            rewards_n = copy.deepcopy(user_feedback_n)

            # save batch result
            full_user_feedback.append(user_feedback_n)

        t2 = time.time()
        print("===============Batch Test===============")
        print("rec model: ", opt.rec_model)
        print("dataset index: ", opt.data_index)
        print("total time cost: %fs" % (t2 - t1))
        print("total recommenda time cost: %fs" % (t_recommend))

        np.savetxt(opt.result_path + opt.rec_model + "/" + opt.res_name +
                   "_batch_data" + str(opt.data_index) +
                   "_full_user_feedback.txt",
                   np.array(full_user_feedback),
                   fmt="%d")
        np.savetxt(
            opt.result_path + opt.rec_model + "/" + opt.res_name +
            "_batch_data" + str(opt.data_index) + "_recommend_time.txt",
            np.array(recommend_time))
        print("*****************total average reward: %4f" %
              (total_average_reward(np.array(full_user_feedback))))
        print("***********************end************************")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('--model',
                        default='SBUCB',
                        choices=["SBUCB", "EXP3_B", "BLTS_B"])
    parser.add_argument('--K',
                        default=10,
                        type=int,
                        help='number of base model and data buffers')
    parser.add_argument('--adao2b', type=int, choices=[0, 1, 2], default=0)
    parser.add_argument('--ckpt_load_name', default='')
    parser.add_argument('--ckpt_save_name', default='')
    parser.add_argument('--data_index', type=int, default=2, choices=[1, 2])
    parser.add_argument('--adao2b_ckpt_name',
                        default='tune',
                        help="adao2b_ckpt_name")
    parser.add_argument('--res_name', default='', help="res_name")
    parser.add_argument('--base_model_save_name', help="base_model_save_name")
    parser.add_argument('--data_selection',
                        help="data_selection",
                        choices=['S', 'R', 'D'],
                        default='S')
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    opt = DefaultConfig(args.model, args.base_model_save_name, args.K,
                        args.data_index)
    opt.rec_model_load_name = args.ckpt_load_name
    opt.rec_model_save_name = args.ckpt_save_name
    opt.adao2b_ckpt_name = args.adao2b_ckpt_name
    opt.res_name = args.res_name
    opt.data_selection = args.data_selection
    opt.seed_num = args.seed

    setup_seed(opt.seed_num)

    if opt.data_selection == 'S':
        opt.base_model_batch_list = [i for i in range(opt.N - opt.K, opt.N)]
    elif opt.data_selection == 'R':
        opt.base_model_batch_list = reservoir_sampling(opt.N, opt.K)
        print("base_model_batch_list: ", opt.base_model_batch_list)
    elif opt.data_selection == 'D':
        opt.base_model_batch_list = data_dependent_sampling(
            opt.history_result_path, opt.history_avg_reward_file_name, opt.B,
            opt.K)

    print('\n'.join(['%s:%s' % item for item in opt.__dict__.items()]))

    # batch_test
    if args.adao2b == 0:
        bacth_test()
    elif args.adao2b == 1:
        ada_bacth_test()
    else:
        batch_size_tune = [512, 1024, 2048]
        lr_tune = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        weight_decay_tune = [1e-6, 1e-5, 1e-4, 1e-3]

        for batch_size, lr, weight_decay in product(batch_size_tune, lr_tune,
                                                    weight_decay_tune):
            print("batch_size, lr, weight_decay:", batch_size, lr,
                  weight_decay)
            opt.adao2b_ckpt_name = args.adao2b_ckpt_name + "_batch_size_" + str(
                batch_size) + "_lr_" + str(lr) + "_weight_decay_" + str(
                    weight_decay)
            opt.res_name = args.res_name + "_batch_size_" + str(
                batch_size) + "_lr_" + str(lr) + "_weight_decay_" + str(
                    weight_decay)
            ada_bacth_test()
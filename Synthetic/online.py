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
from Model.SBUCB import SBUCB
from Model.EXP3_B import EXP3_B
from Model.BLTS_B import BLTS_B
from utils.data import get_candidata_vector
from utils.simulator import get_batch_feedback
from utils.evaluation import total_average_reward
from itertools import product


class DefaultConfig(object):
    def __init__(self, data_index=1):
        self.rec_model = "SBUCB"
        self.device = "cuda:1"

        self.data_index = data_index
        self.feature_dimension = 10
        self.N = 40
        self.B = 5000
        self.candidate_size = 10
        self.ood_num = 20

        # simulator
        self.w = np.random.normal(0.1, 0.01,
                                  (self.feature_dimension, 1))  # reward weight

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


def online_learning():
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

    if opt.rec_model_load_name != '':
        print("****Load model from: ", opt.rec_model_load_name)
        rec_model.load_model(opt.rec_model_load_name)
    else:
        # init batch for synthetic data
        rec_state_n = []
        user_feedback_n = []
        rewards_n = []
        for b in range(opt.B):
            candidate_state_vector_nb = get_candidata_vector(
                opt.data_index, opt.candidate_size, opt.feature_dimension)
            rec_state_nb_index = random.sample(range(opt.candidate_size), 1)[0]
            rec_state_nb = candidate_state_vector_nb[rec_state_nb_index]
            rec_state_n.append(rec_state_nb)

        user_feedback_n = get_batch_feedback(
            np.array(rec_state_n).reshape(opt.B, -1), opt.w)
        rewards_n = copy.deepcopy(user_feedback_n)

        if opt.rec_model == "SBUCB" or opt.rec_model == "EXP3_B" or opt.rec_model == "BLTS_B":
            rec_model.update(
                np.array(rec_state_n).reshape(opt.B, -1),
                np.array(rewards_n).reshape(-1, 1))

    # save result for stastics
    full_user_feedback = []
    data_buffer = []
    t_recommend = 0
    t_offline_training = 0
    recommend_time = []
    online_training_time = []
    offline_training_time = []

    # online learning
    t1 = time.time()
    for n in tqdm(range(opt.N)):
        rec_state_n = []
        user_feedback_n = []
        rewards_n = []

        for b in range(opt.B):
            # Observe the set of candidate items S_𝑛𝑏
            if opt.data_index == 1 and n >= opt.N - opt.ood_num:
                # generate some ood data for data 1 (similiar to data 2)
                candidate_state_vector_nb = get_candidata_vector(
                    3, opt.candidate_size, opt.feature_dimension)
            else:
                candidate_state_vector_nb = get_candidata_vector(
                    opt.data_index, opt.candidate_size, opt.feature_dimension)
            # candidate_state_vector_nb = get_candidata_vector(opt.data_index, opt.candidate_size, opt.feature_dimension)

            # Recommend item 𝒔 ∈ S_𝑛𝑏 to the user
            t_recommend_start = time.time()
            rec_state_nb_index = rec_model.recommend(candidate_state_vector_nb)
            t_recommend_end = time.time()
            t_recommend += t_recommend_end - t_recommend_start
            recommend_time.append(t_recommend_end - t_recommend_start)

            rec_state_nb = candidate_state_vector_nb[rec_state_nb_index]
            rec_state_n.append(rec_state_nb)

            data_buffer_nb = [
                n, candidate_state_vector_nb.tolist(), rec_state_nb_index
            ]
            data_buffer.append(data_buffer_nb)

        # Receive batch user feedback
        user_feedback_n = get_batch_feedback(
            np.array(rec_state_n).reshape(opt.B, -1), opt.w)
        rewards_n = copy.deepcopy(user_feedback_n)

        # save batch result
        full_user_feedback.append(user_feedback_n)

        # save batch model
        # rec_model.save_model(opt.rec_model_save_name + "_batch" + str(n)) # save model before update

        t_offline_training_start = time.time()
        # update rec model
        if opt.rec_model == "SBUCB" or opt.rec_model == "EXP3_B" or opt.rec_model == "BLTS_B":
            rec_model.update(
                np.array(rec_state_n).reshape(opt.B, -1),
                np.array(rewards_n).reshape(-1, 1))

        # save batch model
        rec_model.save_model(opt.rec_model_save_name + "_batch" +
                             str(n))  # save model after update

        t_offline_training_end = time.time()
        t_offline_training += t_offline_training_end - t_offline_training_start
        offline_training_time.append(t_offline_training_end -
                                     t_offline_training_start)

    t2 = time.time()
    print("===============Online Learning===============")
    print("rec model: ", opt.rec_model)
    print("dataset index: ", opt.data_index)
    print("total time cost: %fs" % (t2 - t1))
    print("total recommenda time cost: %fs" % (t_recommend))
    print("total offline training time cost: %fs" % (t_offline_training))

    rec_model.save_model(opt.rec_model_save_name)  # save last model
    np.savetxt(opt.result_path + opt.rec_model + "/" + opt.res_name +
               "_online_data" + str(opt.data_index) +
               "_full_user_feedback.txt",
               np.array(full_user_feedback),
               fmt="%d")
    print("*****************total average reward: %4f" %
          (total_average_reward(np.array(full_user_feedback))))
    print("***********************end************************")
    if opt.rec_model_load_name == '' and opt.data_index == 1:
        data_buffer_df = pd.DataFrame(data_buffer)
        data_buffer_df = pd.concat([
            data_buffer_df,
            pd.DataFrame(np.array(full_user_feedback).reshape(-1))
        ],
                                   axis=1)
        data_buffer_df.columns = [
            "batch", "candidate_vector", "action", "reward"
        ]
        data_buffer_df.to_csv("../Data/AdaO2B/" + opt.rec_model + "_" +
                              opt.res_name + "_online_data" +
                              str(opt.data_index) + "_adao2b.csv",
                              index=False,
                              sep="\t")
    # np.savetxt(
    #     opt.result_path + opt.rec_model + "/" + opt.res_name +
    #     "_online_data" + str(opt.data_index) + "_recommend_time.txt",
    #     np.array(recommend_time))
    # np.savetxt(
    #     opt.result_path + opt.rec_model + "/" + opt.res_name +
    #     "_online_data" + str(opt.data_index) + "_offline_training_time.txt",
    #     np.array(offline_training_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('--model',
                        default='SBUCB',
                        choices=["SBUCB", "EXP3_B", "BLTS_B"])
    parser.add_argument('--ckpt_load_name', default='')
    parser.add_argument('--ckpt_save_name', default='')
    parser.add_argument('--data_index', type=int, default=1, choices=[1, 2])
    parser.add_argument('--res_name',
                        default='',
                        help="version and data information")
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    opt = DefaultConfig(args.data_index)
    opt.rec_model = args.model
    opt.rec_model_load_name = args.ckpt_load_name
    opt.rec_model_save_name = args.ckpt_save_name
    opt.res_name = args.res_name
    opt.seed_num = args.seed

    setup_seed(opt.seed_num)

    print('\n'.join(['%s:%s' % item for item in opt.__dict__.items()]))

    # Online Learning
    online_learning()

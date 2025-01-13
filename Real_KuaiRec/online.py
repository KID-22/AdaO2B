# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import argparse
import random
import copy
import os
import time
from tqdm import tqdm
from Model.SBUCB import SBUCB
from Model.BLTS_B import BLTS_B
from Model.EXP3_B import EXP3_B
from utils.simulator import Simulator
from utils.data import get_online_data, get_candidata_vector
from utils.evaluation import total_average_reward
from itertools import product


class DefaultConfig(object):
    def __init__(self, data_index=1):
        self.rec_model = "SBUCB"
        self.device = "cuda:0"

        self.data_index = data_index
        self.feature_dimension = 50
        self.N = 40
        self.B = 5000
        self.candidate_size = 100

        # simulator
        self.simulator_path = '../Data/Simulator/'
        self.embedding_size = 16
        self.simulator_online_path = self.simulator_path + '1031_simulator.pth'
        self.user_num = 6890
        self.item_num = 973

        # online data
        self.online_path = "../Data/Online/"
        self.user_data_online_path = self.online_path + "user_feature_online.csv"  # 25
        self.item_data_online_path = self.online_path + "item_daily_feature_online.csv"  # 25
        self.session_data_online_path = self.online_path + "data_" + str(
            self.data_index) + "_online.csv"

        # result
        self.result_path = "../Result/"
        self.res_name = ""

        # model save and load
        self.rec_model_load_name = ""
        self.rec_model_save_name = ""

        # SBUCB
        self.SBUCB_mu = 1.4

        # BLTS_B
        self.BLTS_B_mu = 0.2

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
    # load simulator data and model
    online_simulator = Simulator(user_num=opt.user_num,
                                 item_num=opt.item_num,
                                 embedding_size=opt.embedding_size,
                                 simulator_path=opt.simulator_online_path)

    # load online data
    user_feature_online, item_feature_online, session_data_online = get_online_data(
        opt.user_data_online_path, opt.item_data_online_path,
        opt.session_data_online_path)
    uid_online = session_data_online.user_id.values
    date_online = session_data_online.date.values
    candidate_itemid_online = session_data_online.candidate.values

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

    # save result for stastics
    full_user_feedback = []
    data_buffer = []
    t_recommend = 0
    t_offline_training = 0
    recommend_time = []
    online_training_time = []
    offline_training_time = []

    t1 = time.time()
    for n in tqdm(range(opt.N)):
        # print("theta", rec_model.theta)

        u_n = []
        date_n = []
        s_n = []
        user_feedback_n = []
        rewards_n = []

        for b in range(opt.B):
            # Observe the set of candidate items S_𝑛𝑏
            candidate_nb = candidate_itemid_online[n * opt.B + b]
            u_nb = uid_online[n * opt.B + b]
            date_nb = date_online[n * opt.B + b]
            u_n.append(u_nb)
            date_n.append(date_nb)
            candidate_state_vector_nb = get_candidata_vector(
                user_feature_online, item_feature_online, u_nb, candidate_nb,
                date_nb)

            # Recommend item 𝒔 ∈ S_𝑛𝑏 to the user
            t_recommend_start = time.time()
            s_nb_index = rec_model.recommend(candidate_state_vector_nb)
            t_recommend_end = time.time()
            t_recommend += t_recommend_end - t_recommend_start
            recommend_time.append(t_recommend_end - t_recommend_start)

            s_nb = candidate_nb[s_nb_index]
            s_n.append(s_nb)
            data_buffer_nb = [date_nb, n, u_nb, candidate_nb, s_nb_index]
            data_buffer.append(data_buffer_nb)

        # Receive batch user feedback
        user_feedback_n = online_simulator.get_batch_feedback(u_n, s_n)
        rewards_n = copy.deepcopy(user_feedback_n)

        # save batch result
        full_user_feedback.append(user_feedback_n)

        # # save batch model
        # rec_model.save_model(opt.rec_model_save_name + "_batch" + str(n))

        t_offline_training_start = time.time()
        # update rec model
        if opt.rec_model == "SBUCB" or opt.rec_model == "EXP3_B" or opt.rec_model == "BLTS_B":
            user_feature_n = user_feature_online[u_n]
            item_feature_n = []
            for i in range(len(s_n)):
                item_feature_n.append(item_feature_online[date_n[i]][s_n[i]])
            item_feature_n = np.array(item_feature_n)
            state_vector_n = np.concatenate((user_feature_n, item_feature_n),
                                            axis=1)
            rec_model.update(state_vector_n,
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
            "date", "batch", "uid", "candidate", "action", "reward"
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
    parser.add_argument('--seed', type=int, default=2023)

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

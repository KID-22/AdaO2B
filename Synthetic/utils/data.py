from torch.utils import data
import numpy as np
import pandas as pd
import torch
import copy
from torch.utils.data import DataLoader
import random


def get_candidata_vector(data_index, candidate_size, feature_dimension):
    '''
    candidate_size: int, the number of candidate
    feature_dimension: int, the dimension of feature
    return candidate_state_vector: np.array, shape=(candidate_size, feature_dimension)
    '''
    if data_index == 1:
        mu_s = np.linspace(1,-2.6,10)
        sigma_s = 0.05
    elif data_index == 2:
        mu_s = np.linspace(1,-2.6,10)
        mu_s[0] += 0.4
        sigma_s = 0.05
    elif data_index == 3:
        mu_s = np.linspace(1,-2.6,10)
        mu_s[0] += 0.2
        sigma_s = 0.05
    candidate_vector = []
    random.shuffle(mu_s)
    for i in range(candidate_size):
        context_s = np.random.normal(mu_s[i], sigma_s, (feature_dimension))
        candidate_vector.append(context_s) 
    return np.array(candidate_vector)


class Data_4_AdaO2B(data.Dataset):
    def __init__(self, candidate, base_model_rec_score, action, reward):
        self.candidate = candidate
        self.base_model_rec_score = base_model_rec_score
        self.action = action
        self.reward = reward

    def __getitem__(self, index):
        can = self.candidate[index]
        base_score = self.base_model_rec_score[index]
        act = self.action[index]
        rew = self.reward[index]
        can = torch.tensor(can).type(torch.float32)
        base_score = torch.tensor(base_score).type(torch.float32)
        act = torch.tensor(act).type(torch.long)
        rew = torch.tensor(rew).type(torch.float32)

        return can, base_score, act, rew

    def __len__(self):
        return self.reward.shape[0]


def load_base_model(base_model, save_name, batch_list):
    base_model_list = []
    for i in batch_list:
        base_model.load_model(save_name + "_batch" + str(i))
        base_model_list.append(copy.deepcopy(base_model))

    return base_model_list


def get_batch_average_reward(history_result_path, pre_name, batch_list, B):
    user_feedback = np.loadtxt(history_result_path + pre_name + "_full_user_feedback.txt")
    user_feedback = user_feedback.sum(axis=1)
    avg_rw = user_feedback / B
    # print(avg_rw)

    return avg_rw[batch_list]


def get_best_batch(history_result_path, pre_name, B):
    user_feedback = np.loadtxt(history_result_path + pre_name + "_full_user_feedback.txt")
    user_feedback = user_feedback.sum(axis=1)
    avg_rw = user_feedback / B

    return np.argmax(avg_rw)


def reservoir_sampling(N, K):
    '''
        N: the number of data
        K: the number of sample
    '''
    sample = []
    for i in range(N):
        if i < K:
            sample.append(i)
        else:
            j = np.random.randint(0, i)
            if j < K:
                sample[j] = i
    return sample


def data_dependent_sampling(history_result_path, pre_name, B, K):
    user_feedback = np.loadtxt(history_result_path + pre_name + "_full_user_feedback.txt")
    user_feedback = user_feedback.sum(axis=1)
    avg_rw = user_feedback / B

    avg_rw_current = avg_rw[:-1]
    avg_rw_next = avg_rw[1:]
    avg_rw_growth = (avg_rw_next - avg_rw_current) / avg_rw_current 

    # return min K avg_rw_growth index + 1
    return np.argsort(avg_rw_growth)[:K] + 1


def prep_data_4_adao2b(base_model_list, history_data_path, batch_list, all_data=0):
    full_history_data = pd.read_csv(history_data_path, sep="\t")
    if all_data==0:
        history_data = full_history_data[full_history_data["batch"].isin(
            batch_list)]
    else:
        history_data = full_history_data
    history_data["candidate_vector"] = history_data["candidate_vector"].apply(
        lambda x: np.array(eval(x)))
    candidate_state_vector = history_data.candidate_vector.values
    action = history_data.action.values
    reward = history_data.reward.values

    base_rec_score = []
    for tmp_candidate_state_vector in candidate_state_vector:
        tmp_base_rec_score = []
        for base_model in base_model_list:
            tmp_base_rec_score.append(
                base_model.get_rec_score(tmp_candidate_state_vector))
        tmp_base_rec_score = np.array(
            tmp_base_rec_score).T  # (n_candidate, n_base_model)
        base_rec_score.append(tmp_base_rec_score)

    data_4_adao2b = Data_4_AdaO2B(candidate_state_vector, base_rec_score,
                                  action, reward)

    return data_4_adao2b


def get_state_vector_and_reward_for_NIIDC(history_data_path, batch_list, reward=0):
    def get_state_vector(row):
        return row["candidate_vector"][row["action"]]

    full_history_data = pd.read_csv(history_data_path, sep="\t")
    history_data = full_history_data[full_history_data["batch"].isin(batch_list)]
    if reward == 1:
        history_data = history_data[history_data["reward"] == 1]

    history_data["candidate_vector"] = history_data["candidate_vector"].apply(lambda x: np.array(eval(x)))
    history_data["state_vector"] = history_data.apply(get_state_vector, axis=1)

    return np.vstack(history_data["state_vector"].values), history_data["reward"].values

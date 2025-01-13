from torch.utils import data
import numpy as np
import pandas as pd
import torch
import copy
from Model.SBUCB import SBUCB
from Model.BLTS_B import BLTS_B
from Model.EXP3_B import EXP3_B
from torch.utils.data import DataLoader


def get_online_user_item_feature(user_data_online_path, item_data_online_path):
    user_feature_online_df = pd.read_csv(user_data_online_path, sep="\t")
    user_feature_online = user_feature_online_df[[
        "u" + str(i) for i in range(25)
    ]].values
    item_feature_online_df = pd.read_csv(item_data_online_path, sep="\t")
    item_feature_online = {}
    for date in item_feature_online_df["date"].unique():
        date = int(date)
        tmp = {}
        for idx, row in item_feature_online_df[item_feature_online_df["date"]
                                               == date].iterrows():
            video_id = int(row["video_id"])
            tmp[video_id] = row[["v" + str(i) for i in range(25)]].values
        date_video_list = list(tmp.keys())
        for i in set(range(973)) - set(date_video_list):
            tmp[i] = np.zeros(25)
        video_feature = np.array(list(tmp.values()))
        # print(video_feature.shape)
        item_feature_online[date] = video_feature

    return user_feature_online, item_feature_online


def get_online_data(user_data_online_path, item_data_online_path,
                    session_data_online_path):
    user_feature_online, item_feature_online = get_online_user_item_feature(
        user_data_online_path, item_data_online_path)
    session_data_online = pd.read_csv(session_data_online_path, sep="\t")
    session_data_online["user_id"] = session_data_online["user_id"].apply(
        lambda x: int(x))
    session_data_online["date"] = session_data_online["date"].apply(
        lambda x: int(x))
    session_data_online["candidate"] = session_data_online["candidate"].apply(
        lambda x: eval(x))

    return user_feature_online, item_feature_online, session_data_online


def get_candidata_vector(user_feature_online, item_feature_online, u_nb,
                         candidate, date_nb):
    candidate_user_feature = user_feature_online[[u_nb] * len(candidate)]
    candidate_item_feature = item_feature_online[date_nb][candidate]
    candidate_state_vector = np.concatenate(
        (candidate_user_feature, candidate_item_feature), axis=1)

    return candidate_state_vector


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
    user_feedback = np.loadtxt(history_result_path + pre_name +
                               "_full_user_feedback.txt")
    user_feedback = user_feedback.sum(axis=1)
    avg_rw = user_feedback / B
    # print(avg_rw)

    return avg_rw[batch_list]


def get_best_batch(history_result_path, pre_name, B):
    user_feedback = np.loadtxt(history_result_path + pre_name +
                               "_full_user_feedback.txt")
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
    user_feedback = np.loadtxt(history_result_path + pre_name +
                               "_full_user_feedback.txt")
    user_feedback = user_feedback.sum(axis=1)
    avg_rw = user_feedback / B

    avg_rw_current = avg_rw[:-1]
    avg_rw_next = avg_rw[1:]
    avg_rw_growth = (avg_rw_next - avg_rw_current) / avg_rw_current

    # return min K avg_rw_growth index + 1
    return np.argsort(avg_rw_growth)[:K] + 1


def prep_data_4_adao2b(base_model_list,
                       user_data_online_path,
                       item_data_online_path,
                       history_data_path,
                       batch_list,
                       all_data=0):
    user_feature_online, item_feature_online = get_online_user_item_feature(
        user_data_online_path, item_data_online_path)
    full_history_data = pd.read_csv(history_data_path, sep="\t")
    if all_data == 0:
        history_data = full_history_data[full_history_data["batch"].isin(
            batch_list)]
    else:
        history_data = full_history_data
    history_data["candidate"] = history_data["candidate"].apply(
        lambda x: eval(x))
    date = history_data.date.values
    uid = history_data.uid.values
    candidate = history_data.candidate.values
    action = history_data.action.values
    reward = history_data.reward.values

    base_rec_score = []
    candidate_state_vector = []
    for tmp_uid, tmp_candidate, tmp_date in zip(uid, candidate, date):
        tmp_candidate_state_vector = get_candidata_vector(
            user_feature_online, item_feature_online, tmp_uid, tmp_candidate,
            tmp_date)  # n_candidate, n_feature
        tmp_base_rec_score = []
        for base_model in base_model_list:
            tmp_base_rec_score.append(
                base_model.get_rec_score(tmp_candidate_state_vector))
        tmp_base_rec_score = np.array(
            tmp_base_rec_score).T  # (n_candidate, n_base_model)
        # print("-----tmp_base_rec_score", tmp_base_rec_score.shape) # (100,10)
        # print("-----tmp_candidate_state_vector", tmp_candidate_state_vector.shape) # (100,50)
        candidate_state_vector.append(tmp_candidate_state_vector)
        base_rec_score.append(tmp_base_rec_score)

    data_4_adao2b = Data_4_AdaO2B(candidate_state_vector, base_rec_score,
                                  action, reward)

    return data_4_adao2b


def get_state_vector_and_reward_for_NIIDC(user_feature_online,
                                          item_feature_online,
                                          history_data_path,
                                          batch_list,
                                          reward=1):
    def get_selected_itemid(row):
        return row["candidate"][row["action"]]

    full_history_data = pd.read_csv(history_data_path, sep="\t")
    history_data = full_history_data[full_history_data["batch"].isin(
        batch_list)]
    if reward == 1:
        history_data = history_data[history_data["reward"] == 1]

    history_data["candidate"] = history_data["candidate"].apply(
        lambda x: eval(x))
    history_data["selected_itemid"] = history_data.apply(get_selected_itemid,
                                                         axis=1)
    date = history_data.date.values
    uid = history_data.uid.values
    selected_itemid = history_data.selected_itemid.values
    reward = history_data.reward.values

    state_vector = []
    for tmp_uid, tmp_selected_itemid, tmp_date in zip(uid, selected_itemid,
                                                      date):
        user_feature = user_feature_online[tmp_uid]
        item_feature = item_feature_online[tmp_date][tmp_selected_itemid]
        tmp_state_vector = np.hstack((user_feature, item_feature))
        state_vector.append(tmp_state_vector)

    return np.vstack(np.array(state_vector)), reward
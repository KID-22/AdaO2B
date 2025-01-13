import pandas as pd
import numpy as np
import math
import json


class EXP3_B():
    def __init__(self, N=40, B=5000, C=10, delta=0.001, feature_d=10):
        self.phi = np.eye(int(feature_d))
        self.b = np.zeros((int(feature_d), 1))
        self.theta = np.zeros((int(feature_d), 1))
        self.big_theta = []
        self.delta = delta
        self.eta = (2 * (1 - self.delta) * math.log(C) / (C * N * B))**0.5
        self.C = C

    def update(self, state_vector, rewards):
        # state_vector B*d
        # rewards B*1
        self.phi += np.dot(state_vector.T, state_vector)
        self.b += np.dot(state_vector.T, rewards)
        self.theta = np.dot(np.linalg.inv(self.phi), self.b)

        if len(self.big_theta) == 0:
            self.big_theta = self.theta
        else:
            self.big_theta = np.concatenate((self.big_theta, self.theta),
                                            axis=1)  # d*n

    def recommend(self, candidate_vector):
        # candidate_vector C*d
        if len(self.big_theta) == 0:
            pi = np.full((self.C), 1 / self.C)
            candidate_index_list = np.argsort(np.array(pi))[-3:].tolist() # only consider the top 3 candidates
            pi = pi[candidate_index_list] / np.sum(pi[candidate_index_list])
        else:
            rewards_sum = np.sum(np.dot(self.big_theta.T, candidate_vector.T),
                                 axis=0)  # n*d * d*C = n*C => C,
            p = np.exp(self.eta * rewards_sum)  # C,
            q = p / np.sum(p)  # C,
            # print("q:", q)
            pi = (1 - self.delta) * q + self.delta * np.ones(self.C) / self.C # original version
            # candidate_index_list = np.argsort(np.array(q))[-3:].tolist() # only consider the top 3 candidates
            # pi = (1 - self.delta) * q[candidate_index_list]/np.sum(q[candidate_index_list]) + self.delta * np.ones(3) / 3

        # return np.random.choice(candidate_index_list, p=pi)
        return np.argmax(pi)

    def get_rec_score(self, candidate_vector):
        # candidate_vector C*d
        rewards_sum = np.sum(np.dot(self.big_theta.T, candidate_vector.T), axis=0)  # n*d * d*C = n*C => C,
        # p = np.exp(self.eta * rewards_sum)  # C,
        # q = p / np.sum(p)  # C,
        # pi = (1 - self.delta) * q + self.delta * np.ones(self.C) / self.C

        return rewards_sum

    def bt_recommend(self, candidate_vector):
        return np.argmax(self.get_rec_score(candidate_vector))

    def load_model(self, load_path):
        with open("../CKPT/EXP3_B/" + load_path + ".json", 'r') as f:
            model_paras = json.load(f)

        self.phi = np.array(model_paras["phi"])
        self.b = np.array(model_paras["b"])
        self.theta = np.array(model_paras["theta"])
        self.big_theta = np.array(model_paras["big_theta"])
        self.delta = model_paras["delta"]
        self.eta = model_paras["eta"]
        self.C = model_paras["C"]

    def save_model(self, save_path):
        model_paras = {}
        model_paras["phi"] = self.phi.tolist()
        model_paras["b"] = self.b.tolist()
        model_paras["theta"] = self.theta.tolist()
        model_paras["big_theta"] = self.big_theta.tolist()
        model_paras["delta"] = self.delta
        model_paras["eta"] = self.eta
        model_paras["C"] = self.C

        with open("../CKPT/EXP3_B/" + save_path + ".json", 'w') as f:
            json.dump(model_paras, f, indent=4)

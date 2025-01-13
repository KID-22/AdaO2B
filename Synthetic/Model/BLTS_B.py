import pandas as pd
import numpy as np
import json


class BLTS_B():
    def __init__(self, mu=0.4, feature_d=10):
        self.mu = mu
        self.phi = np.eye(feature_d)
        self.phi_inv = np.eye(feature_d)
        self.b = np.zeros((feature_d, 1))
        self.theta = np.zeros((feature_d, 1))
        self.theta_final = np.zeros((feature_d, 1))

    def update(self, state_vector, rewards):
        # state_vector B*d
        # rewards B*1
        self.phi += np.dot(state_vector.T, state_vector)
        self.phi_inv = np.linalg.inv(self.phi)
        self.b += np.dot(state_vector.T, rewards)
        self.theta = np.dot(self.phi_inv, self.b)
        self.theta_final = np.random.multivariate_normal(self.theta.reshape(-1), (self.mu**2) * self.phi_inv, 1).reshape(-1, 1)

    def recommend(self, candidate_vector):
        # candidate_vector C*d
        all_action = np.dot(candidate_vector, self.theta_final).reshape(-1)  # C*d * d*1 = C*1

        return np.argmax(all_action)

    def get_rec_score(self, candidate_vector):
        # score = np.dot(candidate_vector, self.theta_final).reshape(-1)  # C*d * d*1 = C*1

        score = np.dot(candidate_vector, self.theta).reshape(-1)  # C*d * d*1 = C*1

        return score

    def bt_recommend(self, candidate_vector):
        return np.argmax(self.get_rec_score(candidate_vector))

    def load_model(self, load_path):
        with open("../CKPT/BLTS_B/" + load_path + ".json", 'r') as f:
            model_paras = json.load(f)

        self.mu = model_paras["mu"]
        self.phi = np.array(model_paras["phi"])
        self.phi_inv = np.array(model_paras["phi_inv"])
        self.b = np.array(model_paras["b"])
        self.theta = np.array(model_paras["theta"])
        self.theta_final = np.array(model_paras["theta_final"])

    def save_model(self, save_path):
        model_paras = {}
        model_paras["mu"] = self.mu
        model_paras["phi"] = self.phi.tolist()
        model_paras["phi_inv"] = self.phi_inv.tolist()
        model_paras["b"] = self.b.tolist()
        model_paras["theta"] = self.theta.tolist()
        model_paras["theta_final"] = self.theta_final.tolist()

        with open("../CKPT/BLTS_B/" + save_path + ".json", 'w') as f:
            json.dump(model_paras, f, indent=4)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json


class AdaO2B(nn.Module):
    def __init__(self, K, feature_d, EXP3=0):
        super(AdaO2B, self).__init__()
        self.K = K 
        self.feature_d = feature_d
        self.MLP1 = nn.Sequential(nn.Linear(int(self.feature_d), 64),
                                    nn.Tanh(),
                                    nn.Linear(64, 64),
                                    nn.Tanh(),
                                    nn.Linear(64, K))
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.MSELoss()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            nn.init.xavier_normal_(module.weight.data)

    def forward(self, candidate, base_model_rec_score):
        '''
            candidate : batchsize, C, d
            base_model_rec_score : batchsize, C, K
        '''
        ada_weight = self.MLP1(candidate) # batchsize, C, K
        fusion_score = (base_model_rec_score * ada_weight).sum(dim=2) # batchsize, C
        fusion_score_softmax = self.softmax(fusion_score) # batchsize, C

        return fusion_score_softmax

    def calculate_loss(self, candidate, base_model_rec_score, action, reward):
        '''
            candidate : batchsize, C, d
            base_model_rec_score : batchsize, C, K
            action : batchsize
            reward : batchsize
        '''
        action = action.reshape(-1,1) # batchsize, 1
        reward = reward.reshape(-1)
        fusion_score = self.forward(candidate, base_model_rec_score) # batchsize, C
        fusion_action_score = fusion_score.gather(dim=1, index=action).reshape(-1) # batchsize
        return self.loss(fusion_action_score, reward)

    def evaluate(self, candidate, base_model_rec_score, action):
        fusion_score = self.forward(candidate, base_model_rec_score)
        action = action.reshape(-1,1) # batchsize, 1
        fusion_action_score = fusion_score.gather(dim=1, index=action).reshape(-1) # batchsize
        return fusion_action_score

    def recommend(self, candidate, base_model_rec_score):
        return torch.argmax(self.forward(candidate, base_model_rec_score), axis=1) 

    def get_ada_weight(self, candidate):
        return self.MLP1(candidate)

    def get_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)



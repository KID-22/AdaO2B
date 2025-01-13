import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.nn.init import normal_
from scipy.special import expit



class MF(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, device='cpu'):
        super(MF, self).__init__()

        self.num_users = num_users
        self.num_items = num_items

        self.user_e = nn.Embedding(self.num_users, embedding_size)
        self.item_e = nn.Embedding(self.num_items, embedding_size)
        self.user_b = nn.Embedding(self.num_users, 1)
        self.item_b = nn.Embedding(self.num_items, 1)
        self.sigmoid = torch.nn.Sigmoid()

        self.apply(self._init_weights)

        # self.loss = nn.MSELoss()
        self.loss = nn.BCELoss()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.1)

    def forward(self, user, item):
        user_embedding = self.user_e(user)
        item_embedding = self.item_e(item)

        preds = self.user_b(user)
        preds += self.item_b(item)
        preds += (user_embedding * item_embedding).sum(dim=1, keepdim=True)

        return self.sigmoid(preds).squeeze()

    def calculate_loss(self, user_list, item_list, label_list):
        return self.loss(self.forward(user_list, item_list), label_list)

    def predict(self, user, item):
        return self.forward(user, item)

    def get_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.parameters(),
                                lr=lr,
                                weight_decay=weight_decay)

    def get_embedding(self, user, item):
        return self.user_e(user), self.item_e(item)

    @property
    def get_all_embedding(self):
        return self.user_e, self.item_e


class Simulator():
    def __init__(self, user_num, item_num, embedding_size, simulator_path, device='cpu'):
        self.device = device
        self.model = MF(user_num, item_num, embedding_size, self.device)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(simulator_path))

    def get_batch_feedback(self, user, item):
        user = torch.LongTensor(user).to(self.device)
        item = torch.LongTensor(item).to(self.device)
        prediction = self.model.predict(user, item).detach().cpu().numpy()

        # feedback = np.random.binomial(n=1, p=prediction) # feedback ~ Bern(prediction)
        feedback = np.where(prediction > 0.4, 1, 0) # feedback 1 if prediction > 0.4 else 0

        return feedback

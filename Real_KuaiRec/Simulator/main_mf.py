import os
from torch.utils.data import DataLoader
from torch.utils import data
from tqdm import tqdm
from time import time
import numpy as np
import argparse
import random
import torch
import torch.nn as nn
from torch.nn.init import normal_
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import copy
import pandas as pd


class DefaultConfig(object):
    def __init__(self):
        self.model = 'MF'

        self.data = "../Real_KuaiRec/Data/KuaiRec 2.0/processd/full_data.csv"

        self.metric = 'auc'
        self.verbose = 1

        self.option = 0 

        self.device = 'cpu'
        self.batch_size = 512
        self.embedding_size = 16

        self.max_epoch = 20
        self.lr = 0.001
        self.weight_decay = 1e-5

        self.res_name = ""


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


class MF_DATA(data.Dataset):
    def __init__(self, filename):
        data = pd.read_csv(filename, sep="\t")
        raw_matrix = data[["user_id", "video_id", "label"]].values
        self.users_num = int(6890)
        self.items_num = int(973)
        self.data = raw_matrix

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]


def MSE(preds, true):
    squaredError = []
    for i in range(len(preds)):
        dis = true[i] - preds[i]
        squaredError.append(dis * dis)
    return sum(squaredError) / len(squaredError)


def MAE(preds, true):
    absError = []
    for i in range(len(preds)):
        dis = true[i] - preds[i]
        absError.append(abs(dis))
    return sum(absError) / len(absError)


def RMSE(preds, true):
    squaredError = []
    absError = []
    for i in range(len(preds)):
        dis = true[i] - preds[i]
        squaredError.append(dis * dis)
        absError.append(abs(dis))
    from math import sqrt
    return sqrt(sum(squaredError) / len(squaredError))


def AUC(true, preds):
    return roc_auc_score(true, preds)


def evaluate_model(model, val_data, opt):
    true = val_data[:, 2]
    user = torch.LongTensor(val_data[:, 0]).to(opt.device)
    item = torch.LongTensor(val_data[:, 1]).to(opt.device)
    preds = model.predict(user, item)
    # np.savetxt("pred.txt", preds.detach().cpu().numpy())

    mae = MAE(preds, true)
    mse = MSE(preds, true)
    rmse = RMSE(preds, true)
    auc = AUC(true, preds.detach().cpu().numpy())

    for threshold in [0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
        tmp_preds = copy.deepcopy(preds.detach().cpu().numpy())
        tmp_preds[tmp_preds >= threshold] = 1
        tmp_preds[tmp_preds < threshold] = 0
        print("threshold: %.2f" % (threshold))
        print('acc: {0}/{1} = {2}'.format(np.equal(tmp_preds, true).sum(), true.shape[0], np.equal(tmp_preds, true).sum() / true.shape[0]))
        print('absoult auc: {0}'.format(roc_auc_score(true, tmp_preds)))

    return mae, mse, rmse, auc


seed_num = 2022
print("seed_num:", seed_num)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(seed_num)


def train():
    print('train begin')

    all_data = MF_DATA(opt.data)
    if opt.option == 0:
        train_data, val_data = train_test_split(all_data, train_size=0.8, random_state=seed_num)
        train_data = np.array(train_data).astype(int)
        val_data = np.array(val_data).astype(int)
    elif opt.option == 1:
        train_data = all_data

    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True)
    
    # get model
    model = MF(all_data.users_num, all_data.items_num,
               opt.embedding_size, opt.device)
    model.to(opt.device)
    optimizer = model.get_optimizer(opt.lr, opt.weight_decay)

    best_mse = 10000000.
    best_mae = 10000000.
    best_auc = 0
    best_iter = 0

    # train
    model.train()
    for epoch in range(opt.max_epoch):
        t1 = time()
        for i, data in enumerate(train_dataloader):
            user = data[:, 0].to(opt.device)
            item = data[:, 1].to(opt.device)
            label = data[:, 2].to(opt.device)

            loss = model.calculate_loss(user.long(), item.long(),
                                        label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t2 = time()
        print('Epoch %d [%.1f s]:' % (epoch, t2 - t1))
        print('Train Loss = ', loss.item())

        if opt.option == 0:
            (mae, mse, rmse, auc) = evaluate_model(model, val_data, opt)
            if opt.metric == 'mae':
                if mae < best_mae:
                    best_mae, best_mse, best_auc, best_iter = mae, mse, auc, epoch
                    torch.save(model.state_dict(),
                            "./checkpoint/" + opt.res_name + "_mae.pth")
            elif opt.metric == 'mse':
                if mse < best_mse:
                    best_mae, best_mse, best_auc, best_iter = mae, mse, auc, epoch
                    torch.save(model.state_dict(),
                            "./checkpoint/" + opt.res_name + "_mse.pth")
            elif opt.metric == 'auc':
                if auc > best_auc:
                    best_mae, best_mse, best_auc, best_iter = mae, mse, auc, epoch
                    torch.save(model.state_dict(),
                            "./checkpoint/" + opt.res_name + "_auc.pth")


            if epoch % opt.verbose == 0:
                print(
                    'Val MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f [%.1f s]'
                    % (mae, mse, rmse, auc, time() - t2))
                print("------------------------------------------")

    if opt.option == 0:
        print("train end\nBest Epoch %d:  MAE = %.4f, MSE = %.4f, AUC = %.4f" %
            (best_iter, best_mae, best_mse, best_auc))
    elif opt.option == 1:
        torch.save(model.state_dict(), "./checkpoint/" + opt.res_name + ".pth")

    best_model = MF(all_data.users_num, all_data.items_num,
                    opt.embedding_size, opt.device)
    best_model.to(opt.device)

    if opt.option == 0:
        if opt.metric == 'mae':
            best_model.load_state_dict(
                torch.load("./checkpoint/" + opt.res_name + "_mae.pth"))
        elif opt.metric == 'mse':
            best_model.load_state_dict(
                torch.load("./checkpoint/" + opt.res_name + "_mse.pth"))
        elif opt.metric == 'auc':
            best_model.load_state_dict(
                torch.load("./checkpoint/" + opt.res_name + "_auc.pth"))
    elif opt.option == 1:
        best_model.load_state_dict(
            torch.load("./checkpoint/" + opt.res_name + ".pth"))

    print("\n====================== best model ======================")
    mae, mse, rmse, auc = evaluate_model(best_model, train_data, opt)
    print('Train MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f' %
          (mae, mse, rmse, auc))
    if opt.option == 0:
        mae, mse, rmse, auc = evaluate_model(best_model, val_data, opt)
        print('Val MAE = %.4f, MSE = %.4f, RMSE = %.4f, AUC = %.4f' %
            (mae, mse, rmse, auc))
    print("=========================================================\n")

    return best_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('--model', default='MF')
    parser.add_argument('--option', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--metric',
                        default='auc',
                        choices=["mae", "mse", "auc"])
    parser.add_argument('--res_name',
                        default='',
                        help="version and data information")

    args = parser.parse_args()
    opt = DefaultConfig()
    opt.model = args.model
    opt.option = args.option
    opt.batch_size = args.batch_size
    opt.max_epoch = args.epoch
    opt.lr = args.lr
    opt.weight_decay = args.weight_decay
    opt.metric = args.metric
    opt.res_name = args.res_name

    print('\n'.join(['%s:%s' % item for item in opt.__dict__.items()]))

    if opt.model == 'MF':
        best_model = train()
        user_emb, item_emb = best_model.get_all_embedding
        np.savetxt("./checkpoint/" + opt.res_name + "_user_e.txt",
                   user_emb.weight.detach().numpy())
        np.savetxt("./checkpoint/" + opt.res_name + "item_e.txt",
                   item_emb.weight.detach().numpy())

    print('end')
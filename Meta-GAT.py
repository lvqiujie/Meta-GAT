import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from copy import deepcopy, copy
import torch.utils.data as data
from MetaLearner_map import *
from model import Fingerprint
from smiles_feature import *
from dataset import *
import random
import pandas as pd
device = torch.device('cuda:0')

seed = 188
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

### 准备数据迭代器
k_spt = 2 ## support data 的个数

k_spt_pos = 1   # k-shot positive
k_spt_neg = 1   # k-shot negative
k_query = 128  ## query data 的个数
task_num = 9
test_task_num = 3
batch_size = task_num


p_dropout = 0.2
fingerprint_dim = 200
# also known as l2_regularization_lambda
weight_decay = 5
learning_rate = 2.5
# for regression model
output_units_num = 2
radius = 2
T = 2
batchsz = 10
test_batchsz = 20
epochs = 10000

raw_filename = "data/tox21.csv"
tasks = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
              "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5",
                "SR-HSE", "SR-MMP", "SR-p53",]
# raw_filename = "data/sider.csv"
# tasks = ["SIDER1", "SIDER2", "SIDER3", "SIDER4", "SIDER5",
#          "SIDER6", "SIDER7", "SIDER8", "SIDER9", "SIDER10",
#          "SIDER11", "SIDER12", "SIDER13", "SIDER14", "SIDER15",
#          "SIDER16", "SIDER17", "SIDER18", "SIDER19", "SIDER20",
#          "SIDER21", "SIDER22", "SIDER23", "SIDER24", "SIDER25", "SIDER26", "SIDER27"]

smiles_tasks_df = pd.read_csv(raw_filename)
smilesList = smiles_tasks_df.smiles.values
print("number of all smiles: ",len(smilesList))

remained_smiles = []
for smiles in smilesList:
    try:
        remained_smiles.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
    except:
        print(smiles)
        pass
print("number of successfully processed smiles: ", len(remained_smiles))

feature_filename = raw_filename.replace('.csv','.pickle')
if os.path.isfile(feature_filename):
    feature_dicts = pickle.load(open(feature_filename, "rb"))
else:
    feature_dicts = save_smiles_dicts(remained_smiles, feature_filename)

smiles_tasks_df['remained_smiles'] = remained_smiles
remained_df = smiles_tasks_df[smiles_tasks_df["remained_smiles"].isin(feature_dicts['smiles_to_atom_mask'].keys())]
print("number of remained  smiles: ", len(remained_df.smiles.values))

data_train = MyDataset(remained_df, k_spt, k_spt_pos, k_spt_neg, k_query, tasks[:task_num], task_num, batchsz, type="train")
dataset_train = data.DataLoader(dataset=data_train, batch_size=1, shuffle=True)

data_test = MyDataset(remained_df, k_spt, k_spt_pos, k_spt_neg, k_query, tasks[task_num:], test_task_num, test_batchsz, type="test")
dataset_test = data.DataLoader(dataset=data_test, batch_size=1, shuffle=True)


x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(
    [remained_smiles[0]], feature_dicts)
num_atom_features = x_atom.shape[-1]
num_bond_features = x_bonds.shape[-1]

model = Fingerprint(radius, T, num_atom_features, num_bond_features,
                    fingerprint_dim, output_units_num, p_dropout, feature_dicts).to(device)
meta = MetaLearner(model).to(device)


for epoch in range(epochs):

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataset_train):
        accs_train, loss = meta(x_spt, y_spt, x_qry, y_qry)
        # print("@@@@@@@@@@@")
        if step % 100 == 0:
            print("epoch:", epoch, "step:", step)
            print(accs_train)
            # print(loss)

        if step % 1000 == 0:
            accs = [[] for i in range(test_task_num)]
            n = 0
            for x_spt, y_spt, x_qry, y_qry in dataset_test:
                n = n + 1
                # print(n, "###############")
                task_num = len(x_spt)
                shot = len(x_spt[0])
                query_size = len(x_qry[0])
                y_spt = y_spt.view(task_num, shot).long().cuda()
                y_qry = y_qry.view(task_num, query_size).long().cuda()
                for task_index, (x_spt_one, y_spt_one, x_qry_one, y_qry_one) in enumerate(zip(x_spt, y_spt, x_qry, y_qry)):
                    test_acc = meta.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one, task_index)
                    accs[task_index].append(test_acc)
            accs_res = np.array(accs).mean(axis=1).astype(np.float16)
            print('测试集准确率:', accs_res)



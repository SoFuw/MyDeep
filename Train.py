from multiprocessing.dummy import active_children
from pandas import RangeIndex
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.MYDataSet_MLP import KeyPointDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import json
import argparse
import time
import matplotlib.pyplot as plt
import hashlib
import json
from os import listdir
from os.path import isfile, join
import pandas as pd
import seaborn as sns
from copy import deepcopy
from utils.MyMLP import MLP, args, MyNomalizer, center_of_position_nomalizer, center_of_position_nomalizer_with_normalize
## --Random Seed Initialization --##
seed = 666
np.random.seed(seed)
torch.manual_seed(seed)


# ====== Experiment Variable ====== #
name_var1 = 'n_layers'
name_var2 = 'lr'
list_var1 = [1]
list_var2 = [0.001]


## -- Load datatset --##
trainset = KeyPointDataset("Train")
vaildationset = KeyPointDataset("Validate")
testset = KeyPointDataset("Test")
partition = {'train': trainset, 'val': vaildationset, 'test': testset}


def metric(y_pred, y_true):
    val, indices = torch.max(y_pred, dim=1)
    count = 0
    for i, j in zip(range(y_true.size(dim=0)), indices):
        if(y_true[i][j] == 1.0):

            count += 1

    return count/y_true.size(dim=0)

##사용하지 않음##
# def metric2(y_pred,y_true):
#     count=0
#     data,indices=torch.min((y_pred-1.0).abs(),dim=1)
#     for i,idx in enumerate(indices):
#         if(y_true[i][idx]==1.0):
#             count+=1
#     return count/y_true.size(dim=0)
###############


def train(model, partition, optim, loss_fn):
    model.train()
    model.zero_grad()
    optim.zero_grad()

    trainloader = DataLoader(
        partition['train'], batch_size=args.batch_size, shuffle=True, drop_last=True)

    train_acc = 0.0
    train_loss = 0.0

    for i, data in enumerate(trainloader):
        input = MyNomalizer(data[0].to(args.device), args.device)
        y_pred = model(input)
        y_true = data[1].to(args.device)
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        optim.step()
        train_loss += loss.item()
        train_acc += metric(y_pred, y_true)

    train_loss = train_loss/len(trainloader)
    train_acc = train_acc/len(trainloader)

    return model, train_loss, train_acc


def validate(model, partition, loss_fn, args):
    val_loader = DataLoader(
        partition['val'], batch_size=args.batch_size, shuffle=False, drop_last=True)
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            input = MyNomalizer(data[0].to(args.device), args.device)
            y_pred = model(input)
            y_true = data[1].to(args.device)
            loss = loss_fn(y_pred, y_true)
            val_loss += loss.item()
            val_acc += metric(y_pred, y_true)

    val_loss = val_loss/len(val_loader)
    val_acc = val_acc/len(val_loader)

    return val_loss, val_acc


def test(model, partition, loss_fn, args):
    test_loder = DataLoader(partition['test'],
                            batch_size=args.batch_size,
                            shuffle=False, drop_last=True)
    test_acc = 0.0
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(test_loder):
            input = MyNomalizer(data[0].to(args.device), args.device)
            y_pred = model(input)
            y_true = data[1].to(args.device)
            test_acc += metric(y_pred, y_true)
    test_acc = test_acc/len(test_loder)
    return test_acc


def save_exp_result(setting, result):
    exp_name = setting['exp_name']
    del setting['epoch']
    hash_key = hashlib.sha1(str(setting).encode()).hexdigest()[:6]
    filename = './results_MLP/{}-{}.json'.format(exp_name, hash_key)
    result.update(setting)
    with open(filename, 'w') as f:
        json.dump(result, f)


def load_exp_result(exp_name):
    dir_path = './results_MLP'
    filenames = [f for f in listdir(dir_path) if isfile(
        join(dir_path, f)) if '.json' in f]
    list_result = []
    for filename in filenames:
        if exp_name in filename:
            with open(join(dir_path, filename), 'r') as infile:
                results = json.load(infile)
                list_result.append(results)
    df = pd.DataFrame(list_result)  # .drop(columns=[])
    return df


def experiment(partition, args):
    model = MLP(args)
    model.to(args.device)
    loss_fn = nn.CrossEntropyLoss()

    if args.optim == 'SGD':
        optimizer = optim.RMSprop(
            model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(
            model.parameters(), lr=args.lr, weight_decay=args.l2)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.l2)
    else:
        raise ValueError('In-valid optimizer choice')

      # ===== List for epoch-wise data ====== #
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    # ===================================== #

    for epoch in range(args.epoch):
        ts = time.time()
        model, train_loss, train_acc = train(
            model, partition, optimizer, loss_fn)
        val_loss, val_acc = validate(model, partition, loss_fn, args)
        te = time.time()
        # ====== Add Epoch Data ====== #
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        # ============================ #
        if((epoch+1) % 50 == 0):
           torch.save(model.state_dict(), join(
               "models", f'model_{args.lr}_{args.n_layers}_{epoch+1}.pt'))
        print('Epoch {}, Acc(train/val): {:2.2f}/{:2.2f}, Loss(train/val) {:2.5f}/{:2.5f}. Took {:2.2f} sec'.format(
            epoch, train_acc, val_acc, train_loss, val_loss, te-ts))

    test_acc = test(model, partition, loss_fn, args)
    print(f'test_acc : {test_acc}')
    # ======= Add Result to Dictionary ======= #
    result = {}
    result['train_losses'] = train_losses
    result['val_losses'] = val_losses
    result['train_accs'] = train_accs
    result['val_accs'] = val_accs
    result['train_acc'] = train_acc
    result['val_acc'] = val_acc
    result['test_acc'] = test_acc
    return vars(args), result


for var1 in list_var1:
    for var2 in list_var2:
        setattr(args, name_var1, var1)
        setattr(args, name_var2, var2)
        print(args)

        setting, result = experiment(partition, deepcopy(args))
        save_exp_result(setting, result)

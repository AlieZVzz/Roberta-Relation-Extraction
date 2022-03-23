#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       : main function to carry out training and testing
@Author             : Kevinpro
@version            : 1.0
'''
import numpy as np
import logging
import time
from sklearn.preprocessing import MinMaxScaler
import time
import copy
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
import warnings
import torch
import time
import argparse
import json
import os
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification
from tqdm import tqdm

# from transformers import BertPreTrainedModel


from transformers import BertModel

from loader import load_train
from loader import load_dev

from loader import map_id_rel
import random


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


time_stamp = time.strftime("%m-%d-%H-%M", time.localtime())
logger = get_logger('log/Roberta-RI' + str(time_stamp) + '.log')


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # torch.save(model.state_dict(), 'checkpoint.pt')	# 这里会存储迄今最优模型的参数
        torch.save(model.state_dict(), 'checkpoints/' + 'Roberta-best_params-' + str(time_stamp) + '.pkl')
        self.val_loss_min = val_loss


early_stopping = EarlyStopping(5, verbose=True)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


setup_seed(2021)

rel2id, id2rel = map_id_rel()

logger.info(len(rel2id))
logger.info(id2rel)


def get_model():
    labels_num = len(rel2id)
    from model import BERT_Classifier
    model = BERT_Classifier(labels_num)
    return model


model = get_model()
# torch.save(model, './bert-base-chinese/test'+'.pth')
# exit()
# exit()
USE_CUDA = torch.cuda.is_available()
# USE_CUDA=False

data = load_train()
logger.info(len(data['text']))
train_text = data['text']
train_mask = data['mask']
train_label = data['label']

train_text = [t.numpy() for t in train_text]
train_mask = [t.numpy() for t in train_mask]

train_text = torch.tensor(train_text)
train_mask = torch.tensor(train_mask)
train_label = torch.tensor(train_label)

# logger.info("--train data--")
# logger.info(train_text.shape)
# logger.info(train_mask.shape)
# logger.info(train_label.shape)

data = load_dev()
dev_text = data['text']
dev_mask = data['mask']
dev_label = data['label']

dev_text = [t.numpy() for t in dev_text]
dev_mask = [t.numpy() for t in dev_mask]

dev_text = torch.tensor(dev_text)
dev_mask = torch.tensor(dev_mask)
dev_label = torch.tensor(dev_label)

logger.info("--train data--")
logger.info(train_text.shape)
logger.info(train_mask.shape)
logger.info(train_label.shape)

logger.info("--eval data--")
logger.info(dev_text.shape)
logger.info(dev_mask.shape)
logger.info(dev_label.shape)

# exit()
# USE_CUDA=False

if USE_CUDA:
    logger.info("using GPU")

train_dataset = torch.utils.data.TensorDataset(train_text, train_mask, train_label)
dev_dataset = torch.utils.data.TensorDataset(dev_text, dev_mask, dev_label)


def eval(net, dataset, batch_size):
    net.eval()
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
    with torch.no_grad():
        correct = 0
        total = 0
        iter = 0
        total_y = []
        total_pred = []
        for text, mask, y in tqdm(train_iter):
            iter += 1
            if text.size(0) != batch_size:
                break
            text = text.reshape(batch_size, -1)
            mask = mask.reshape(batch_size, -1)

            if USE_CUDA:
                text = text.cuda()
                mask = mask.cuda()
                y = y.cuda()

            outputs = net(text, mask, y)
            # logger.info(y)
            loss, logits = outputs[0], outputs[1]
            _, predicted = torch.max(logits.data, 1)
            total += text.size(0)
            total_y.extend(y.cpu())
            total_pred.extend(predicted.cpu().numpy().tolist())
            correct += predicted.data.eq(y.data).cpu().sum()
            s = ("Acc:%.3f" % ((1.0 * correct.numpy()) / total))
        acc = (1.0 * correct.numpy()) / total
        logger.info(
            "Eval Result: right {} total, {} total, Acc: {} ".format(correct.cpu().numpy().tolist(), total, acc))
        logger.info(classification_report(y_true=total_y, y_pred=total_pred))
        return acc, loss


def train(net, dataset, num_epochs, learning_rate, batch_size):
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    param_optimizer = list(net.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.05},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=10, num_training_steps=len(train_iter) * num_epochs)
    # optimizer = AdamW(net.parameters(), lr=learning_rate)

    pre = 0

    for epoch in range(num_epochs):
        correct = 0
        total = 0
        iter = 0
        net.train()

        for text, mask, y in tqdm(train_iter):
            iter += 1
            net.zero_grad()
            # print(type(y))
            # print(y)
            if text.size(0) != batch_size:
                break
            text = text.reshape(batch_size, -1)
            mask = mask.reshape(batch_size, -1)
            if USE_CUDA:
                text = text.cuda()
                mask = mask.cuda()
                y = y.cuda()
            # print(text.shape)
            loss, logits = net(text, mask, y)
            # print(y)
            # print(loss.shape)
            # print("predicted",predicted)
            # print("answer", y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # print(outputs[1].shape)
            # print(output)
            # print(outputs[1])
            _, predicted = torch.max(logits.data, 1)
            total += text.size(0)
            correct += predicted.data.eq(y.data).cpu().sum()
        loss = loss.detach().cpu()
        # print("epoch ", str(epoch), " loss: ", loss.mean().numpy().tolist(), "right",
        #             correct.cpu().numpy().tolist(),
        #             "total", total, "Acc:", correct.cpu().numpy().tolist() / total)
        logger.info('epoch {} loss {} right {} total {} acc {}'.format(str(epoch), loss.mean().numpy().tolist(),
                                                                       correct.cpu().numpy().tolist(), total,
                                                                       correct.cpu().numpy().tolist() / total))

        acc, eval_loss = eval(net, dev_dataset, 32)
        early_stopping(eval_loss, net)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            # 结束模型训练
            break
    return


# model=nn.DataParallel(model,device_ids=[0,1])
if USE_CUDA:
    model = model.cuda()

train(model, train_dataset, 30, 4e-5, 32)
# eval(model,dev_dataset,8)

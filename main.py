# coding: utf-8
import os
import time
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torch.autograd import Variable
from lr_scheduler import *
from model import *
from data import *

import tensorboardX
import tensorflow as tf
from tensorboardX import SummaryWriter

if (__name__ == '__main__'):    
    SEED = 1
    torch.manual_seed(SEED) #保证随机初始化的状态相同
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    opt = __import__('options')


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def data_loader(data_path, batch_size, workers):
    dsets = MyDataset(data_path)
    dsets_loaders = torch.utils.data.DataLoader(dsets, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    dsets_size = len(dsets)
    print('\nStatistics: {}:{}'.format(data_path, dsets_size))
    return dsets_loaders, dsets_size

dset_train_loaders, dsets_train_sizes = data_loader(opt.train_path, opt.batch_size, opt.workers)
dset_dev_loaders, dsets_dev_sizes = data_loader(opt.dev_path, opt.batch_size, opt.workers)
dset_tst_loaders, dsets_tst_sizes = data_loader(opt.tst_path, opt.batch_size, opt.workers)
#shiyan_loaders, dsets_sizes = data_loader(opt.shiyan_path, opt.batch_size, opt.workers)

if (__name__ == '__main__'):
    model = Phone_classify(13, 39, 39, 78, opt.n_classes).cuda()

    writer = SummaryWriter()

    if (hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-6)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

    iteration = 0
    n = 0

    for epoch in range(opt.epochs):
        start_time = time.time()
        exp_lr_scheduler.step()

        for i, batch in enumerate(dset_train_loaders):

            inputs, targets = Variable(batch['audio'].cuda()), Variable(batch['target'].cuda())
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()

            writer.add_scalar('data/scalar_loss_ce', loss.detach().cpu().numpy(), iteration)

            iteration += 1

            loss.backward()
            optimizer.step()

            train_loss = loss.item()

            print('iteration:%d, epoch:%d, train_loss:%.6f'%(iteration, epoch, train_loss))

        if (epoch % 1) == 0:
                corrects = 0
                all_data = 0
                acc_sample = 0
                with torch.no_grad():
                    for idx, batch in enumerate(dset_dev_loaders):
                        inputs, targets = Variable(batch['audio'].cuda()), Variable(batch['target'].cuda())
                        outputs = model(inputs)
                        _, preds = torch.max(F.softmax(outputs, dim=1).data, 1)
                        corrects += torch.sum(preds == targets.data)
                        all_data += len(inputs)
                        acc_sample += torch.sum(preds == targets.data)/len(inputs)
                         
                acc_frame = corrects.cpu().numpy() / all_data
                acc_sample = acc_sample / 62237
                acc_sample = acc_sample.cpu().numpy()
                writer.add_scalar('data/scalar_acc_frame', acc_frame, n)
                writer.add_scalar('data/scalar_acc_sample', acc_sample, n)
                #print('iteration:%d, epoch:%d, acc:%d' % (iteration, epoch, acc/opt.batch_size))

                savename = os.path.join(opt.savedir, 'iteration_{}_epoch_{}_accframe_{}_accsamples_{}.pt'.format(iteration, epoch, acc_frame, acc_sample))
                savepath = os.path.split(savename)[0]
                if (not os.path.exists(savepath)):os.makedirs(savepath)
                #torch.save(model.state_dict(), savename)
                if acc_frame >=0.85 or acc_sample >=0.80:torch.save(model.state_dict(), savename)
                n += 1
            
    
    writer.close()
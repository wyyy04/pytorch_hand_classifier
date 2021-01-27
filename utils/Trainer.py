from __future__ import print_function

import os
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torchnet import meter
import torch
import time

from .log import logger
# from .visualize import Visualizer


def get_learning_rates(optimizer):
    lrs = [pg['lr'] for pg in optimizer.param_groups]
    lrs = np.asarray(lrs, dtype=np.float)
    return lrs


def evaluate(score,ground_truth):
    score = np.array((score))
    ground_truth = np.array(ground_truth)
    img_num,moti_num = score.shape
    score = pd.DataFrame(score)
    score = score.rank(ascending=False,method='first', axis=1)
    score = np.array(score)
    res = score[range(img_num), ground_truth]
    # res = np.median(res)
    return res

class TrainParams(object):
    '''
    Params of the model
    '''

    # required params
    max_epoch = 30

    # optimizer and criterion and learning rate scheduler
    optimizer = None
    criterion = None
    lr_scheduler = None         # should be an instance of ReduceLROnPlateau or _LRScheduler

    # params based on your local env
    gpus = []  # default to use CPU mode
    save_dir = './models/'            # default `save_dir`

    # loading existing checkpoint
    ckpt = None                 # path to the ckpt file

    # saving checkpoints
    save_freq_epoch = 1         # save one ckpt per `save_freq_epoch` epochs


class Trainer(object):

    TrainParams = TrainParams

    def __init__(self, model, train_params, train_data, val_data=None):
        assert isinstance(train_params, TrainParams)
        self.params = train_params

        # Data loaders
        self.train_data = train_data
        self.val_data = val_data

        # criterion and Optimizer and learning rate
        self.last_epoch = 0
        self.criterion = self.params.criterion
        self.optimizer = self.params.optimizer
        self.lr_scheduler = self.params.lr_scheduler
        logger.info('Set criterion to {}'.format(type(self.criterion)))
        logger.info('Set optimizer to {}'.format(type(self.optimizer)))
        logger.info('Set lr_scheduler to {}'.format(type(self.lr_scheduler)))

        # load model
        self.model = model
        logger.info('Set output dir to {}'.format(self.params.save_dir))
        if os.path.isdir(self.params.save_dir):
            pass
        else:
            os.makedirs(self.params.save_dir)

        ckpt = self.params.ckpt
        if ckpt is not None:
            self._load_ckpt(ckpt)
            logger.info('Load ckpt from {}'.format(ckpt))

        # set CUDA_VISIBLE_DEVICES
        if len(self.params.gpus) > 0:
            gpus = ','.join([str(x) for x in self.params.gpus])
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
            self.params.gpus = tuple(range(len(self.params.gpus)))
            logger.info('Set CUDA_VISIBLE_DEVICES to {}...'.format(gpus))
            self.model = nn.DataParallel(self.model, device_ids=self.params.gpus)
            self.model = self.model.cuda()

        self.model.train()

    def train(self):
        # vis = Visualizer()
        best_epoch = 0
        best_loss = np.inf
        best_mr = 256

        filename = os.path.join(self.params.save_dir, 'resnet101_motivation_params.pkl')
        try:
            self.model.load_state_dict(torch.load(filename))
            print("Model exists.")
        except:
            print("No model exists.")

        self.model.train()
        for epoch in range(self.params.max_epoch):

            self.last_epoch += 1
            epoch_start = time.time()
            logger.info('Start training epoch {}'.format(self.last_epoch))

            # train & test for a epoch
            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc, val_mr = self._val_one_epoch()

            # save model
            if best_mr > val_mr:
                best_mr = val_mr
                best_epoch = epoch + 1
                filename = os.path.join(self.params.save_dir, 'resnet101_motivation_params.pkl')
                torch.save(self.model.state_dict(), filename)
            # if (self.last_epoch % self.params.save_freq_epoch == 0) or (self.last_epoch == self.params.max_epoch - 1):
            #     save_name = os.path.join(self.params.save_dir, 'ckpt_epoch_{}.pth'.format(self.last_epoch))
            #     t.save(self.model.state_dict(), save_name)

            if val_loss < best_loss:
                # logger.info('Found a better ckpt ({:.3f} -> {:.3f}), '.format(best_loss, self.loss_meter.value()[0]))
                best_loss = val_loss

            # visualize
            # vis.plot('loss', self.loss_meter.value()[0])
            # vis.plot('val_accuracy', val_accuracy)
            # vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            #     epoch=epoch, loss=self.loss_meter.value()[0], val_cm=str(val_cm.value()),
            #     train_cm=str(self.confusion_matrix.value()), lr=get_learning_rates(self.optimizer)))
            epoch_end = time.time()
            logger.info("Epoch:{epoch},lr:{lr}".format(epoch=epoch, lr=get_learning_rates(self.optimizer)))
            logger.info(
                "\tTraining: Loss: {}, Accuracy: {:.4f}%, \n"
                "\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Median Rank: {}\n"
                "\tTime: {:.4f}s"
                    .format(
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    val_mr,
                    epoch_end - epoch_start))
            logger.info("Best median rank for test : {:.4f} at epoch {:03d}\n".format(best_mr, best_epoch))

            # adjust the lr
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(val_loss, self.last_epoch)

    def _load_ckpt(self, ckpt):
        self.model.load_state_dict(t.load(ckpt))

    def _train_one_epoch(self):
        self.model.train()
        for step, (data, label) in enumerate(self.train_data):

            # meters
            loss_meter = meter.AverageValueMeter()
            confusion_matrix = meter.ConfusionMeter(256)

            inputs = Variable(data)
            target = Variable(label)
            if len(self.params.gpus) > 0:
                inputs = inputs.cuda()
                target = target.cuda()

            # forward
            score = self.model(inputs)
            loss = self.criterion(score, target)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step(None)

            # meters update
            loss_meter.add(loss.data)
            # print(score.data,target.data)
            confusion_matrix.add(score.data, target.data)

            train_loss = loss_meter.mean
            cm_value = confusion_matrix.value()
            train_acc = 100. * np.trace(cm_value) / (cm_value.sum())

            print("Train_Loss:{},Train_Acc:{}".format(train_loss,train_acc))

            return train_loss, train_acc

    def _val_one_epoch(self):

        gt_rank = []

        loss_meter = meter.AverageValueMeter()
        confusion_matrix = meter.ConfusionMeter(256)
        logger.info('Val on validation set...')

        self.model.eval()
        for step, (data, label) in enumerate(self.val_data):
            if step>10 :break
            # val model
            with torch.no_grad():
                inputs = Variable(data)
                target = Variable(label)
            if len(self.params.gpus) > 0:
                inputs = inputs.cuda()
                target = target.cuda()

            score = self.model(inputs)
            loss = self.criterion(score, target)
            loss_meter.add(loss.data)
            confusion_matrix.add(score.data.squeeze(), target.data)

            gt_rank.append(evaluate(score.cpu().detach().numpy(), target.cpu().detach().numpy()))


        test_loss = loss_meter.mean
        cm_value = confusion_matrix.value()
        test_acc = 100. * np.trace(cm_value) / (cm_value.sum())

        test_mr = np.median(np.array(gt_rank).reshape(-1))

        print("Test_Loss:{},Test_Acc:{},Test_mr:{}".format(test_loss, test_acc, test_mr))

        return test_loss, test_acc, test_mr

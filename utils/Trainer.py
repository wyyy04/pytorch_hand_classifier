from __future__ import print_function

import os
import numpy as np
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

        # meters
        self.loss_meter = meter.AverageValueMeter()
        self.confusion_matrix = meter.ConfusionMeter(256)

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
        best_loss = np.inf
        for epoch in range(self.params.max_epoch):

            epoch_start = time.time()
            train_loss = 0.0
            train_acc = 0.0
            test_loss = 0.0
            test_acc = 0.0
            res = np.zeros((len(self.val_data), 256))
            print("len:",len(self.val_data))

            self.loss_meter.reset()
            self.confusion_matrix.reset()

            self.last_epoch += 1
            logger.info('Start training epoch {}'.format(self.last_epoch))

            self._train_one_epoch()

            # save model
            if (self.last_epoch % self.params.save_freq_epoch == 0) or (self.last_epoch == self.params.max_epoch - 1):
                save_name = os.path.join(self.params.save_dir, 'ckpt_epoch_{}.pth'.format(self.last_epoch))
                t.save(self.model.state_dict(), save_name)

            val_cm, val_accuracy = self._val_one_epoch()

            if self.loss_meter.value()[0] < best_loss:
                logger.info('Found a better ckpt ({:.3f} -> {:.3f}), '.format(best_loss, self.loss_meter.value()[0]))
                best_loss = self.loss_meter.value()[0]

            # visualize
            # vis.plot('loss', self.loss_meter.value()[0])
            # vis.plot('val_accuracy', val_accuracy)
            # vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            #     epoch=epoch, loss=self.loss_meter.value()[0], val_cm=str(val_cm.value()),
            #     train_cm=str(self.confusion_matrix.value()), lr=get_learning_rates(self.optimizer)))
            epoch_end = time.time()
            logger.info("Epoch:{epoch},lr:{lr}".format(epoch=epoch, lr=get_learning_rates(self.optimizer)))
            print(
                "\tTraining: Loss: {:.4f}, Accuracy: {:.4f}%, \nValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Median Rank: {} \nTime: {:.4f}s".format(
                    avg_test_loss, avg_train_acc * 100, avg_test_loss, avg_test_acc * 100, mean_rank,
                    epoch_end - epoch_start
                ))
            print("Best median rank for test : {:.4f} at epoch {:03d}".format(best_mr, best_epoch))

            # adjust the lr
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(self.loss_meter.value()[0], self.last_epoch)

    def _load_ckpt(self, ckpt):
        self.model.load_state_dict(t.load(ckpt))

    def _train_one_epoch(self):
        for step, (data, label) in enumerate(self.train_data):
            # train model
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
            self.loss_meter.add(loss.data)
            # print(score.data,target.data)
            self.confusion_matrix.add(score.data, target.data)

            if step > 5: break

    def _val_one_epoch(self):
        self.model.eval()
        confusion_matrix = meter.ConfusionMeter(256)
        logger.info('Val on validation set...')

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
            confusion_matrix.add(score.data.squeeze(), target.data)
            # confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))

        self.model.train()
        cm_value = confusion_matrix.value()

        accuracy = 100. * np.trace(cm_value) / (cm_value.sum())
        return confusion_matrix, accuracy

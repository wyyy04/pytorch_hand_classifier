import logging


def get_logger(name='root'):
    formatter = logging.Formatter(
        # fmt='%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s')
        fmt='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger



logger = get_logger('root')


# a = 0
# logger.info('Set criterion to {}'.format(type(a)))
#
# logger.info('loss', self.loss_meter.value()[0])
# vis.plot('val_accuracy', val_accuracy)
# vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
#     epoch=epoch, loss=self.loss_meter.value()[0], val_cm=str(val_cm.value()),
#     train_cm=str(self.confusion_matrix.value()), lr=get_learning_rates(self.optimizer)))




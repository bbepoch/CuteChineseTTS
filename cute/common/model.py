################################################################################
# Author: BigBangEpoch <bbepoch@163.com>
# Date  : 2018-12-24
# Copyright (c) 2018-2019 BigBangEpoch All rights reserved.
################################################################################


import numpy as np
import time
import cntk
from cntk.layers import Sequential, Recurrence, Dense, LSTM


def fold_once(x, n_fold=4):
    num_frame, dim_x = x.shape[0:2]
    if num_frame % n_fold == 0:
        n_add = 0
        x = x.reshape(-1, dim_x*n_fold)
    else:
        n_add = n_fold - num_frame % n_fold
        vec_add = np.zeros([n_add, dim_x], dtype=np.float32)
        x = np.concatenate((x, vec_add), axis=0).reshape(-1, dim_x*n_fold)
    return x, n_add


def fold_batch(xs, n_fold=4):
    xs_new, n_adds = [], []
    for x in xs:
        x_new, n_add = fold_once(x, n_fold)
        xs_new.append(x_new)
        n_adds.append(n_add)
    return xs_new, n_adds


def loss_fun(output, label):
    length = cntk.sequence.reduce_sum(cntk.reduce_sum(output) * 0 + 1)
    return cntk.sequence.reduce_sum(cntk.reduce_sum(cntk.square(output - label))) / length


class CuteModel(object):
    def __init__(self, dim_x, dim_y):
        self.dim_x = int(dim_x)
        self.dim_y = int(dim_y)
        self.input = cntk.sequence.input_variable(shape=(self.dim_x, ))
        self.label = cntk.sequence.input_variable(shape=(self.dim_y, ))
        self.output = self.model(self.input)
        self.loss = loss_fun(self.output, self.label)
        self.eval = loss_fun(self.output, self.label)
        self.learner = cntk.momentum_sgd(parameters=self.output.parameters,
                                         momentum=cntk.momentum_schedule(0.5),
                                         lr=cntk.learning_rate_schedule(0.006, cntk.UnitType.sample))
        self.trainer = cntk.Trainer(self.output, (self.loss, self.eval), [self.learner])

    def model(self, x):
        param1 = 500
        param2 = 250
        x = Dense(param1, activation=cntk.tanh)(x)
        x = Dense(param1, activation=cntk.tanh)(x)
        x = Dense(param1, activation=cntk.tanh)(x)
        x = Sequential([(Recurrence(LSTM(param2)), Recurrence(LSTM(param2), go_backwards=True)), cntk.splice])(x)
        x = Sequential([(Recurrence(LSTM(param2)), Recurrence(LSTM(param2), go_backwards=True)), cntk.splice])(x)
        x = Dense(self.dim_y)(x)
        return x


def test(n_fold=4):
    input_xs = [np.empty([922, 93], dtype=np.float32)]
    input_xs, _ = fold_batch(xs=input_xs, n_fold=n_fold)
    cntk.device.try_set_default_device(cntk.cpu())
    nn_model = CuteModel(dim_x=93*n_fold, dim_y=199*n_fold)
    t1 = time.time()
    output = nn_model.trainer.model.eval(input_xs)
    print(output[0].shape, time.time()-t1)


if __name__ == '__main__':
    test(n_fold=1)

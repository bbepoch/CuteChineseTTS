################################################################################
# Author: BigBangEpoch <bbepoch@163.com>
# Date  : 2018-12-24
# Copyright (c) 2018-2019 BigBangEpoch All rights reserved.
################################################################################


from cute.common.parameter import parameter_manager
from cute.common.model import fold_batch
from cute.common.model import CuteModel
from cute.train.loader import DataLoader
import numpy as np
import datetime
import time
import os
import cntk


def train(data_file_list, num_train, num_valid):
    n_fold, dim_x, dim_y = 1, 93, 199

    # prepare training data
    train_file_list = data_file_list[0:num_train]
    assert len(data_file_list) >= num_train + num_valid
    valid_file_list = data_file_list[num_train:num_train+num_valid]
    train_data_reader = DataLoader(file_list=train_file_list, dim_x=dim_x, dim_y=dim_y, batch_size=4, shuffle=True)
    valid_data_reader = DataLoader(file_list=valid_file_list, dim_x=dim_x, dim_y=dim_y, batch_size=4, shuffle=False)

    # configure trainer
    nn_model = CuteModel(dim_x=dim_x*n_fold, dim_y=dim_y*n_fold)

    for epoch in range(50):
        begin_time = time.time()
        log_file = open(parameter_manager.training_los_log, 'a')

        # # update learning rate if possible
        if epoch in [12]:
            learning_rate = nn_model.learner.learning_rate() * 0.5
            nn_model.learner.reset_learning_rate(cntk.learning_rate_schedule(learning_rate, cntk.UnitType.sample))

        # train
        train_loss_list = []
        while not train_data_reader.is_finish():
            xx_train, yy_train = train_data_reader.load_one_batch()
            xx_train = fold_batch(xx_train, n_fold=n_fold)[0]
            yy_train = fold_batch(yy_train, n_fold=n_fold)[0]
            nn_model.trainer.train_minibatch({nn_model.input: xx_train, nn_model.label: yy_train})
            curr_train_loss = nn_model.trainer.previous_minibatch_loss_average
            train_loss_list.append(curr_train_loss)
        train_data_reader.reset()

        # validation
        valid_loss_list = []
        while not valid_data_reader.is_finish():
            xx_valid, yy_valid = valid_data_reader.load_one_batch()
            xx_valid = fold_batch(xx_valid, n_fold=n_fold)[0]
            yy_valid = fold_batch(yy_valid, n_fold=n_fold)[0]
            curr_valid_loss = nn_model.trainer.test_minibatch({nn_model.input: xx_valid, nn_model.label: yy_valid})
            valid_loss_list.append(curr_valid_loss)
        valid_data_reader.reset()

        # save check point
        nn_model.trainer.save_checkpoint(parameter_manager.saved_ckp_prefix + str(epoch))

        # calculate loss
        curr_valid_loss = np.mean(np.asarray(valid_loss_list)).item()
        curr_train_loss = np.mean(np.asarray(train_loss_list)).item()

        curr_time = datetime.datetime.now().strftime('%Y-%m-%d %T')
        message = 'Epoch:%04d, Lr:%f, Train:%f, Valid:%f, Time:%f, Time:%s' \
                  % (epoch, nn_model.learner.learning_rate(), curr_train_loss,
                     curr_valid_loss, time.time()-begin_time, curr_time)
        log_file.write(message+'\n')
        log_file.close()
        print(message)


def main():
    # set device
    cntk.device.try_set_default_device(cntk.gpu(0))
    # set data paths
    data_file_list = [os.path.join(parameter_manager.norm_file_folder, line.strip())
                      for line in open(parameter_manager.norm_file_sheets, 'r').readlines()]
    # train the model
    num_train = parameter_manager.num_train
    num_valid = parameter_manager.num_valid
    train(data_file_list=data_file_list, num_train=num_train, num_valid=num_valid)


if __name__ == '__main__':
    main()

################################################################################
# Author: BigBangEpoch <bbepoch@163.com>
# Date  : 2018-12-24
# Copyright (c) 2018-2019 BigBangEpoch All rights reserved.
################################################################################


def de_normalize(feat_data, norm_params, norm_type='mean_var'):
    """
    de-normalize data with normalization parameters using norm_type defined method
    :param feat_data: data to de-normalize
    :param norm_params: a numpy array of shape (4, N), indicating min, max, mean and variance for N dimensions
    :param norm_type: str type, 'min_max' or 'mean_var'
    :return:
        numpy array, sharing same shape with input data
    """
    assert feat_data.shape[1] == norm_params.shape[1]
    assert norm_type in ['min_max', 'mean_var']
    if norm_type == 'min_max':
        min_val, min_target = norm_params[0], 0.01
        max_val, max_target = norm_params[1], 0.99
        return (max_val - min_val + 0.001) * (feat_data - min_target) / (max_target - min_target) + min_val
    else:
        mean_val = norm_params[2]
        variance = norm_params[3]
        return feat_data * variance + mean_val



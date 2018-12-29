################################################################################
#           The Neural Network (NN) based Speech Synthesis System
#                https://svn.ecdf.ed.ac.uk/repo/inf/dnn_tts/
#
#                Centre for Speech Technology Research
#                     University of Edinburgh, UK
#                      Copyright (c) 2014-2015
#                        All Rights Reserved.
#
# The system as a whole and most of the files in it are distributed
# under the following copyright and conditions
#
#  Permission is hereby granted, free of charge, to use and distribute
#  this software and its documentation without restriction, including
#  without limitation the rights to use, copy, modify, merge, publish,
#  distribute, sublicense, and/or sell copies of this work, and to
#  permit persons to whom this work is furnished to do so, subject to
#  the following conditions:
#
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   - The authors' names may not be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK
#  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
#  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT
#  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
#  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
#  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
#  THIS SOFTWARE.
################################################################################
# Source: Modified from open source code
# Author: BigBangEpoch <bbepoch@163.com>
# Date  : 2018-12-24
# Copyright (c) 2018-2019 BigBangEpoch All rights reserved.
################################################################################


from cute.common.parameter import parameter_manager
from tqdm import tqdm
import numpy as np
import os


def interpolate_f0(lf0):
    vuv = np.zeros(shape=lf0.shape, dtype=np.float32)
    vuv[lf0 > 0.0] = 1.0
    num_frame, last_value, i = lf0.shape[0], 0.0, 0
    lf0_new = np.zeros(shape=lf0.shape, dtype=np.float32)
    while i < num_frame:
        if lf0[i] <= 0.0:
            j = i + 1
            for j in range(i + 1, num_frame):
                if lf0[j] > 0.0:
                    break
            if j < num_frame - 1:
                if last_value > 0.0:
                    step = (lf0[j] - lf0[i - 1]) / float(j - i)
                    for k in range(i, j):
                        lf0_new[k] = lf0[i - 1] + step * (k - i + 1)
                else:
                    for k in range(i, j):
                        lf0_new[k] = lf0[j]
            else:
                for k in range(i, num_frame):
                    lf0_new[k] = last_value
            i = j
        else:
            lf0_new[i] = lf0[i]
            last_value = lf0[i]
            i += 1
    return lf0_new, vuv


def weighted_sum(feat, weighted_win):
    assert len(weighted_win) > 1 and len(weighted_win) % 2 == 1
    win_width = len(weighted_win) // 2
    feat_new = np.zeros(shape=feat.shape, dtype=np.float32)
    for col in range(feat.shape[1]):
        tmp = np.pad(feat[:, col], (win_width, win_width), 'edge')
        for i in range(feat.shape[0]):
            for w in range(len(weighted_win)):
                feat_new[i, col] += tmp[i+w] * weighted_win[w]
    return feat_new


def extend_to_dynamic(feat):
    feat_org = feat
    feat_dlt = weighted_sum(feat=feat_org, weighted_win=np.array([-0.5, 0.0, 0.5], dtype=np.float32))
    feat_acc = weighted_sum(feat=feat_org, weighted_win=np.array([1.0, -2.0, 1.0], dtype=np.float32))
    return np.concatenate([feat_org, feat_dlt, feat_acc], axis=1)


def feat_statistics(feat):
    min_col_val = np.min(feat, axis=0)
    max_col_val = np.max(feat, axis=0)
    sum_col_val = np.sum(feat, axis=0)
    return np.vstack([min_col_val, max_col_val, sum_col_val])


def normalize_feature(feat, norm_params, dim_lab=260, norm_type='mean_var'):
    if norm_type not in ['min_max', 'mean_var']:
        raise ValueError
    if norm_type == 'mean_var':
        mean_val = norm_params[2]
        variance = norm_params[3]
        feat_new = (feat - mean_val) / variance
    else:
        min_val, min_target = norm_params[0], 0.01
        max_val, max_target = norm_params[1], 0.99
        feat_new = (feat - min_val) * (max_target - min_target) / (max_val - min_val + 0.001) + min_target
    # do not normalize feature lab
    feat_new[:, 0:dim_lab] = feat[:, 0:dim_lab]
    return feat_new


def merge_features(lab_path, mgc_path, lf0_path, bap_path, out_path, dim_lab=260, dim_mgc=60, dim_lf0=1, dim_bap=5):
    lab = np.fromfile(lab_path, dtype=np.float32).reshape(-1, dim_lab).astype(np.float32)
    mgc = np.fromfile(mgc_path, dtype=np.float32).reshape(-1, dim_mgc).astype(np.float32)
    lf0 = np.fromfile(lf0_path, dtype=np.float32).reshape(-1, dim_lf0).astype(np.float32)
    bap = np.fromfile(bap_path, dtype=np.float32).reshape(-1, dim_bap).astype(np.float32)
    assert mgc.shape[0] == lf0.shape[0] and mgc.shape[0] == bap.shape[0]
    # assert abs(lab.shape[0] - mgc.shape[0]) <= 1
    if abs(lab.shape[0] - mgc.shape[0]) <= 1:
        num_frame = min(lab.shape[0], mgc.shape[0])
        lab = lab[0:num_frame, :]
        lf0, vuv = interpolate_f0(lf0=lf0[0:num_frame, :])
        lf0 = extend_to_dynamic(feat=lf0)
        mgc = extend_to_dynamic(feat=mgc[0:num_frame, :])
        bap = extend_to_dynamic(feat=bap[0:num_frame, :])
        merged_feat = np.concatenate([lab, mgc, lf0, vuv, bap], axis=1)
        merged_feat.tofile(out_path)
        min_max_sum = feat_statistics(feat=merged_feat)
        return num_frame, min_max_sum
    else:
        print('merge feature: invalid', lab.shape[0], mgc.shape[0], lab_path)
        return None, None


def normalize_all_file(lab_folder, aco_folder, num_files=4000, feat_type='two_hot'):
    assert feat_type in ['one_hot', 'two_hot']
    dim_lab = 260 if feat_type == 'one_hot' else 93

    tmp_feature_folder = os.path.join(aco_folder, 'tmp')
    if not os.path.exists(tmp_feature_folder):
        os.mkdir(tmp_feature_folder)
    # merge multiple feature files into one
    min_max_mean_var, num_samples = None, 0
    for k in tqdm(range(num_files)):
        file_id = '%06d' % (k+1)  # '000001'
        lab_path = os.path.join(lab_folder, file_id + '.lab')
        mgc_path = os.path.join(aco_folder, 'mgc', file_id + '.mgc')
        lf0_path = os.path.join(aco_folder, 'lf0', file_id + '.lf0')
        bap_path = os.path.join(aco_folder, 'bap', file_id + '.bap')
        tmp_path = os.path.join(aco_folder, 'tmp', file_id + '.tmp')
        num_frame, min_max_sum = merge_features(lab_path=lab_path, mgc_path=mgc_path, lf0_path=lf0_path,
                                                bap_path=bap_path, out_path=tmp_path, dim_lab=dim_lab)
        if num_frame is None:
            continue
        # find min and max of feat
        if min_max_mean_var is None:
            min_max_mean_var = min_max_sum
        else:
            min_val = np.min(np.vstack([min_max_sum[0, :], min_max_mean_var[0, :]]), axis=0)
            max_val = np.max(np.vstack([min_max_sum[1, :], min_max_mean_var[1, :]]), axis=0)
            sum_val = np.sum(np.vstack([min_max_sum[2, :], min_max_mean_var[2, :]]), axis=0)
            min_max_mean_var = np.vstack([min_val, max_val, sum_val])
        num_samples += num_frame
    assert min_max_mean_var is not None and num_samples != 0

    # calculate mean of feat and add empty row for variance
    min_max_mean_var[2, :] = min_max_mean_var[2, :] / num_samples
    min_max_mean_var = np.vstack([min_max_mean_var,
                                  np.zeros(shape=(min_max_mean_var.shape[1], ), dtype=np.float32)])

    # calculate variance of feat
    for k in tqdm(range(num_files)):
        file_id = '%06d' % (k + 1)
        in_file_path = os.path.join(tmp_feature_folder, file_id + '.tmp')
        if not os.path.exists(in_file_path):
            print('calculate var: invalid', in_file_path)
            continue
        feat = np.fromfile(in_file_path, dtype=np.float32).reshape(-1, min_max_mean_var.shape[1])
        min_max_mean_var[3, :] = min_max_mean_var[3, :] + np.sum((feat - min_max_mean_var[2, :]) ** 2, axis=0)
    min_max_mean_var[3, :] = (min_max_mean_var[3, :] / num_samples) ** 0.5
    min_max_mean_var.tofile(os.path.join(aco_folder, 'features.norm'))

    # normalize features with min max method
    norm_features_folder = os.path.join(aco_folder, 'norm')
    if not os.path.exists(norm_features_folder):
        os.mkdir(norm_features_folder)
    for k in tqdm(range(num_files)):
        file_id = '%06d' % (k + 1)
        in_file_path = os.path.join(tmp_feature_folder, file_id + '.tmp')
        if not os.path.exists(in_file_path):
            print('normalization: invalid', in_file_path)
            continue
        feat = np.fromfile(in_file_path, dtype=np.float32).reshape(-1, min_max_mean_var.shape[1])
        feat_new = normalize_feature(feat=feat, norm_params=min_max_mean_var, dim_lab=dim_lab)
        norm_file_path = os.path.join(norm_features_folder, file_id + '.nor')
        feat_new.tofile(norm_file_path)


def main():
    lab_files_folder = parameter_manager.lab_files_folder
    aco_feats_folder = parameter_manager.aco_feats_folder
    num_acos_to_norm = parameter_manager.num_acos_to_norm
    normalize_all_file(lab_folder=lab_files_folder, aco_folder=aco_feats_folder, num_files=num_acos_to_norm)


if __name__ == '__main__':
    main()

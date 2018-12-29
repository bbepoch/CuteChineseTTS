################################################################################
# Author: BigBangEpoch <bbepoch@163.com>
# Date  : 2018-12-24
# Copyright (c) 2018-2019 BigBangEpoch All rights reserved.
################################################################################


from cute.common.converter import pinyin_to_feature_matrix as pinyin2feat
from cute.common.normalizer import de_normalize
from cute.common.vocoder import generate_wave
from cute.common.model import fold_once
from cute.common.model import CuteModel
import numpy as np
import cntk
import os


def setup_nn_model(model_path, dim_input=93, dim_output=199, n_fold=1):
    cntk.device.try_set_default_device(cntk.cpu())
    nn_model = CuteModel(dim_x=dim_input*n_fold, dim_y=dim_output*n_fold)
    nn_model.trainer.restore_from_checkpoint(model_path)
    return nn_model.trainer.model


def predict_acoustics(nn_predictor, feat_lab, norm_params, n_fold=1):
    dim_mgc, dim_lf0, dim_vuv, dim_bap = 60, 1, 1, 5
    # fold input feature
    feat_new, n_add = fold_once(feat_lab, n_fold=n_fold)
    # forward and de-normalize
    predicted_data = nn_predictor.eval(feat_new)[0]
    # unfold output feature
    predicted_data = predicted_data.reshape(predicted_data.shape[0]*n_fold, -1)
    if n_add != 0:
        predicted_data = predicted_data[:-n_add]
    mlb = de_normalize(feat_data=predicted_data, norm_params=norm_params)
    # split all kinds of features: mgc, lf0, vuv, bap
    mgc = mlb[:, 0:dim_mgc * 3]
    lf0 = mlb[:, dim_mgc * 3: dim_mgc * 3 + dim_lf0 * 3]
    vuv = mlb[:, dim_mgc * 3 + dim_lf0 * 3: dim_mgc * 3 + dim_lf0 * 3 + dim_vuv]
    bap = mlb[:, dim_mgc * 3 + dim_lf0 * 3 + dim_vuv: dim_mgc * 3 + dim_lf0 * 3 + dim_vuv + dim_bap * 3]
    print(mlb.shape, mgc.shape, lf0.shape, vuv.shape, bap.shape)
    # extract static part
    mgc = mgc[:, 0:dim_mgc]
    lf0 = lf0[:, 0:dim_lf0]
    vuv = vuv[:, 0:dim_vuv]
    bap = bap[:, 0:dim_bap]
    lf0[vuv < 0.5] = -1.0e+10
    return mgc, lf0, bap


def acoustic_to_audio(mgc, lf0, bap, output_file_path):
    # save mgc, lf0, bap to file
    mgc_file_path = output_file_path + '.mgc'
    lf0_file_path = output_file_path + '.lf0'
    bap_file_path = output_file_path + '.bap'
    mgc.tofile(mgc_file_path)
    lf0.tofile(lf0_file_path)
    bap.tofile(bap_file_path)
    # synthesize waveform
    generate_wave(mgc_file_path=mgc_file_path, do_post_filter=True)
    os.system('rm -f {mgc} {lf0} {bap}'.format(mgc=mgc_file_path, lf0=lf0_file_path, bap=bap_file_path))


def obtain_paths():
    norm_params_path = '../cute/pretrained/features.norm'
    model_path = '../cute/pretrained/model_26'
    assert os.path.exists(norm_params_path), norm_params_path
    assert os.path.exists(model_path), model_path
    return norm_params_path, model_path


def batch_test():
    norm_params_path, model_path = obtain_paths()
    min_max_mean_var = np.fromfile(norm_params_path, dtype=np.float32).reshape(4, -1)[:, 93:]
    nn_model = setup_nn_model(model_path=model_path)
    # feature order in norm file: lab, mgc, lf0, vuv, bap
    dim_lab, dim_mgc, dim_lf0, dim_vuv, dim_bap = 93, 60, 1, 1, 5
    dim_ttl = 93 + 60 * 3 + 1 * 3 + 1 + 5 * 3
    for i in range(10):
        norm_file_path = '../cute/test_data/nn/%06d.nor' % (3600+i)
        assert os.path.exists(norm_file_path), norm_file_path
        feat_lab = np.fromfile(norm_file_path, np.float32).reshape(-1, dim_ttl)[:, 0:dim_lab]
        mgc, lf0, bap = predict_acoustics(nn_predictor=nn_model, feat_lab=feat_lab, norm_params=min_max_mean_var)
        output_file_path = norm_file_path[0:-4]
        acoustic_to_audio(mgc=mgc, lf0=lf0, bap=bap, output_file_path=output_file_path)


def pipeline(pinyin_str):
    norm_params_path, model_path = obtain_paths()
    min_max_mean_var = np.fromfile(norm_params_path, dtype=np.float32).reshape(4, -1)[:, 93:]
    nn_model = setup_nn_model(model_path=model_path)
    # pinyin to audio
    feat_lab = pinyin2feat(pinyin_str=pinyin_str, show=True)
    mgc, lf0, bap = predict_acoustics(nn_predictor=nn_model, feat_lab=feat_lab, norm_params=min_max_mean_var)
    output_file_path = '../cute/test_data/result'
    acoustic_to_audio(mgc=mgc, lf0=lf0, bap=bap, output_file_path=output_file_path)


if __name__ == '__main__':
    """
    pinyin to audio: change to your input pinyin here
    """
    # batch_test()
    input_pinyin = 'ai4 yin1 si1 tan3 pei2 wai4 sun1 wan2 shou3 ji1'
    pipeline(pinyin_str=input_pinyin)

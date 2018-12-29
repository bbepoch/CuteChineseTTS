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
import numpy as np
import subprocess
import os


world_dir = parameter_manager.bin_world_path
sp_tk_dir = parameter_manager.bin_sp_tk_path


WORLD = {
    'ANALYSIS': os.path.join(world_dir, 'analysis'),
    'SYNTHESIS': os.path.join(world_dir, 'synth'),
}

SPTK = {
    'X2X': os.path.join(sp_tk_dir, 'x2x'),
    'BCP': os.path.join(sp_tk_dir, 'bcp'),
    'MLPG': os.path.join(sp_tk_dir, 'mlpg'),
    'VSUM': os.path.join(sp_tk_dir, 'vsum'),
    'SOPR': os.path.join(sp_tk_dir, 'sopr'),
    'VOPR': os.path.join(sp_tk_dir, 'vopr'),
    'MC2B': os.path.join(sp_tk_dir, 'mc2b'),
    'B2MC': os.path.join(sp_tk_dir, 'b2mc'),
    'MERGE': os.path.join(sp_tk_dir, 'merge'),
    'VSTAT': os.path.join(sp_tk_dir, 'vstat'),
    'FREQT': os.path.join(sp_tk_dir, 'freqt'),
    'C2ACR': os.path.join(sp_tk_dir, 'c2acr'),
    'MGC2SP': os.path.join(sp_tk_dir, 'mgc2sp'),
}


def run_process(args):
    p = None
    try:
        # buf size=-1 enables buffering and may improve performance compared to the unbuffered case
        # better to use communicate() than read() and write() - this avoids deadlocks
        p = subprocess.Popen(args, bufsize=-1, shell=True, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
        std_out_data, std_err_data = p.communicate()
        if p.returncode != 0:
            print('exit status: %d' % p.returncode)
            print('for command: %s' % args)
            print('     stderr: %s' % std_err_data)
            print('     stdout: %s' % std_out_data)
            raise OSError
        return std_out_data, std_err_data
    except subprocess.CalledProcessError as e:
        print('exit status: %d' % e.returncode)
        print('for command: %s' % args)
        print('     output: %s' % e.output)
        raise
    except ValueError:
        print('ValueError for %s' % args)
        raise
    except OSError:
        print('OSError for %s' % args)
        raise
    except KeyboardInterrupt:
        print('KeyboardInterrupt during %s' % args)
        try:
            if p is not None:
                p.kill()
        except UnboundLocalError:
            pass
        raise KeyboardInterrupt


def bark_alpha(sr):
    return 0.8517 * np.sqrt(np.arctan(0.06583 * sr / 1000.0)) - 0.1916


def erb_alpha(sr):
    return 0.5941 * np.sqrt(np.arctan(0.1418 * sr / 1000.0)) + 0.03237


def post_filter(mgc_file_path, dim_mgc=60, pf_coef=1.2, co_coef=1023, fl_coef=2048, fw_coef=0.766):
    if not os.path.exists(mgc_file_path):
        raise FileNotFoundError
    tmp_weight_file = mgc_file_path + '_weight'
    tmp_mgc_r0_file = mgc_file_path + '_r0'
    tmp_mgc_b0_file = mgc_file_path + '_b0'
    tmp_mgc_pr_file = mgc_file_path + '_p_r0'
    tmp_mgc_pb_file = mgc_file_path + '_p_b0'
    tmp_mgc_pf_file = mgc_file_path + '_p_mgc'

    line = ''.join(['echo 1 1 '] + ['%.3f ' % pf_coef] * (dim_mgc - 2))
    run_process('{line} | {x2x} +af > {weight}'.format(line=line, x2x=SPTK['X2X'], weight=tmp_weight_file))

    run_process('{freqt} -m {order} -a {fw} -M {co} -A 0 < {mgc} | {c2acr} -m {co} -M 0 -l {fl} > {base_r0}'
                .format(freqt=SPTK['FREQT'], order=dim_mgc - 1, fw=fw_coef, co=co_coef, mgc=mgc_file_path,
                        c2acr=SPTK['C2ACR'], fl=fl_coef, base_r0=tmp_mgc_r0_file))

    run_process('{vopr} -m -n {order} < {mgc} {weight} | {freqt} -m {order} -a {fw} -M {co} -A 0 | {c2acr} '
                '-m {co} -M 0 -l {fl} > {base_p_r0}'
                .format(vopr=SPTK['VOPR'], order=dim_mgc - 1, mgc=mgc_file_path, weight=tmp_weight_file,
                        freqt=SPTK['FREQT'], fw=fw_coef, co=co_coef, c2acr=SPTK['C2ACR'], fl=fl_coef,
                        base_p_r0=tmp_mgc_pr_file))

    run_process('{vopr} -m -n {order} < {mgc} {weight} | {mc2b} -m {order} -a {fw} | {bcp} -n {order} '
                '-s 0 -e 0 > {base_b0}'
                .format(vopr=SPTK['VOPR'], order=dim_mgc - 1, mgc=mgc_file_path, mc2b=SPTK['MC2B'], fw=fw_coef,
                        weight=tmp_weight_file, bcp=SPTK['BCP'], base_b0=tmp_mgc_b0_file))

    run_process('{vopr} -d < {base_r0} {base_p_r0} | {sopr} -LN -d 2 | {vopr} -a {base_b0} > {base_p_b0}'
                .format(vopr=SPTK['VOPR'], base_r0=tmp_mgc_r0_file, base_p_r0=tmp_mgc_pr_file,
                        sopr=SPTK['SOPR'], base_b0=tmp_mgc_b0_file, base_p_b0=tmp_mgc_pb_file))

    run_process('{vopr} -m -n {order} < {mgc} {weight} | {mc2b} -m {order} -a {fw} | {bcp} -n {order} -s 1 '
                '-e {order} | {merge} -n {order2} -s 0 -N 0 {base_p_b0} | {b2mc} -m {order} -a {fw} > {base_p_mgc}'
                .format(vopr=SPTK['VOPR'], order=dim_mgc - 1, mgc=mgc_file_path, weight=tmp_weight_file,
                        mc2b=SPTK['MC2B'], fw=fw_coef, bcp=SPTK['BCP'], merge=SPTK['MERGE'], order2=dim_mgc - 2,
                        base_p_b0=tmp_mgc_pb_file, b2mc=SPTK['B2MC'], base_p_mgc=tmp_mgc_pf_file))

    run_process('rm -f {r0} {b0} {pr0} {pb0} {weight}'
                .format(r0=tmp_mgc_r0_file, b0=tmp_mgc_b0_file, pr0=tmp_mgc_pr_file, pb0=tmp_mgc_pb_file,
                        weight=tmp_weight_file))
    return tmp_mgc_pf_file


def generate_wave(mgc_file_path, do_post_filter=True):
    base_path = mgc_file_path[:-4]
    lf0_file_path = base_path + '.lf0'
    bap_file_path = base_path + '.bap'
    assert os.path.exists(mgc_file_path), mgc_file_path
    assert os.path.exists(lf0_file_path), lf0_file_path
    assert os.path.exists(bap_file_path), bap_file_path

    sample_rate, dim_mgc, pf_coef, co_coef, fl_coef = 48000, 60, 1.5, 1023, 2048
    fw_coef = bark_alpha(sr=sample_rate)

    tmp_ap_file = base_path + '.ap'
    tmp_sp_file = base_path + '.sp'
    tmp_f0_file = base_path + '.f0'
    output_wave = base_path + '.wav'

    # post filter
    if do_post_filter:
        mgc_file_path = post_filter(mgc_file_path=mgc_file_path, dim_mgc=dim_mgc,
                                    pf_coef=pf_coef, co_coef=co_coef, fl_coef=fl_coef, fw_coef=fw_coef)

    # synthesize waveform
    run_process('{sopr} -magic -1.0E+10 -EXP -MAGIC 0.0 {lf0} | {x2x} +fd > {f0}'
                .format(sopr=SPTK['SOPR'], lf0=lf0_file_path, x2x=SPTK['X2X'], f0=tmp_f0_file))

    run_process('{sopr} -c 0 {bap} | {x2x} +fd > {ap}'
                .format(sopr=SPTK['SOPR'], bap=bap_file_path, x2x=SPTK['X2X'], ap=tmp_ap_file))

    run_process('{mgc2sp} -a {alpha} -g 0 -m {order} -l {fl} -o 2 {mgc} | {sopr} -d 32768.0 -P | {x2x} +fd > {sp}'
                .format(mgc2sp=SPTK['MGC2SP'], alpha=fw_coef, order=dim_mgc-1, fl=fl_coef, mgc=mgc_file_path,
                        sopr=SPTK['SOPR'], x2x=SPTK['X2X'], sp=tmp_sp_file))

    run_process('{synworld} {fl} {sr} {f0} {sp} {ap} {wav}'
                .format(synworld=WORLD['SYNTHESIS'], fl=fl_coef, sr=sample_rate, f0=tmp_f0_file, sp=tmp_sp_file,
                        ap=tmp_ap_file, wav=output_wave))

    run_process('rm -f {ap} {sp} {f0}'.format(ap=tmp_ap_file, sp=tmp_sp_file, f0=tmp_f0_file))
    if do_post_filter and mgc_file_path.endswith('.mgc_p_mgc'):
        run_process('rm -f {mgc_pf}'.format(mgc_pf=mgc_file_path))
    os.system('play {wav}'.format(wav=output_wave))


def test():
    generate_wave('../../cute/test_data/extracted/test.mgc', do_post_filter=True)


if __name__ == '__main__':
    test()

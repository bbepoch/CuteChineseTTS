#!/bin/sh

# speech tools directory
# speech_tools_dir="~/program/speech-tools"
# input audio directory
# wav_dir="~/DATA/audio/baker/48k/wav"
# Output features directory
# out_dir="~/DATA/audio/baker/48k/features"

if [ $# != 3 ]
then
echo "invalid arguments"
exit 1
fi

speech_tools_dir=$1
wav_dir=$2
out_dir=$3

# tools directory
world="${speech_tools_dir}/bin/WORLD"
sptk="${speech_tools_dir}/bin/SPTK-3.9"

sp_dir="${out_dir}/sp"
mgc_dir="${out_dir}/mgc"
ap_dir="${out_dir}/ap"
bap_dir="${out_dir}/bap"
f0_dir="${out_dir}/f0"
lf0_dir="${out_dir}/lf0"
resyn_dir="${out_dir}/resyn_dir"

mkdir -p ${out_dir}
mkdir -p ${sp_dir}
mkdir -p ${mgc_dir}
mkdir -p ${bap_dir}
mkdir -p ${f0_dir}
mkdir -p ${lf0_dir}
mkdir -p ${resyn_dir}

# initializations
fs=48000

if [ "${fs}" -eq 16000 ]
then
nFFTHalf=1024 
alpha=0.58
fi

if [ "${fs}" -eq 22050 ]
then
nFFTHalf=1024
alpha=0.65
fi

if [ "${fs}" -eq 44100 ]
then
nFFTHalf=2048
alpha=0.76
fi

if [ "${fs}" -eq 48000 ]
then
nFFTHalf=2048
alpha=0.77
fi

# bap order depends on sampling freq.
mc_size=59

for file in ${wav_dir}/*.wav #.wav
do
    filename="${file##*/}"
    file_id="${filename%.*}"
    echo ${file_id}

    ### WORLD ANALYSIS -- extract vocoder parameters ###

    ### extract f0, sp, ap ### 
    ${world}/analysis ${wav_dir}/${file_id}.wav ${f0_dir}/${file_id}.f0 ${sp_dir}/${file_id}.sp ${bap_dir}/${file_id}.bapd

    ### convert f0 to lf0 ###
    ${sptk}/x2x +da ${f0_dir}/${file_id}.f0 > ${f0_dir}/${file_id}.f0a
    ${sptk}/x2x +af ${f0_dir}/${file_id}.f0a | ${sptk}/sopr -magic 0.0 -LN -MAGIC -1.0E+10 > ${lf0_dir}/${file_id}.lf0
    
    ### convert sp to mgc ###
    ${sptk}/x2x +df ${sp_dir}/${file_id}.sp | ${sptk}/sopr -R -m 32768.0 | ${sptk}/mcep -a ${alpha} -m ${mc_size} -l ${nFFTHalf} -e 1.0E-8 -j 0 -f 0.0 -q 3 > ${mgc_dir}/${file_id}.mgc

    ### convert bapd to bap ###
    ${sptk}/x2x +df ${bap_dir}/${file_id}.bapd > ${bap_dir}/${file_id}.bap

#    ### WORLD Re-synthesis -- reconstruction of parameters ###
#
#    ### convert lf0 to f0 ###
#    ${sptk}/sopr -magic -1.0E+10 -EXP -MAGIC 0.0 ${lf0_dir}/${file_id}.lf0 | ${sptk}/x2x +fa > ${resyn_dir}/${file_id}.resyn.f0a
#    ${sptk}/x2x +ad ${resyn_dir}/${file_id}.resyn.f0a > ${resyn_dir}/${file_id}.resyn.f0
#
#    ### convert mgc to sp ###
#    ${sptk}/mgc2sp -a ${alpha} -g 0 -m ${mc_size} -l ${nFFTHalf} -o 2 ${mgc_dir}/${file_id}.mgc | ${sptk}/sopr -d 32768.0 -P | ${sptk}/x2x +fd > ${resyn_dir}/${file_id}.resyn.sp
#
#    ### convert bapd to bap ###
#    ${sptk}/x2x +fd ${bap_dir}/${file_id}.bap > ${resyn_dir}/${file_id}.resyn.bapd
#
#    ${world}/synth ${nFFTHalf} ${fs} ${resyn_dir}/${file_id}.resyn.f0 ${resyn_dir}/${file_id}.resyn.sp ${resyn_dir}/${file_id}.resyn.bapd ${out_dir}/${file_id}.resyn.wav
done

rm -rf ${sp_dir} ${ap_dir} ${f0_dir} ${bap_dir}/*.bapd
rm -rf ${resyn_dir}

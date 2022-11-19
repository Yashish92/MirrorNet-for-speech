#!/usr/bin/env python
# coding: utf-8
'''
This script performs audio segmentation, generates auditory spectrograms and create the final train, test files

Author : Yashish

'''
import matplotlib

SERVER = True  # This variable enable or disable matplotlib, set it to true when you use the server!
if SERVER:
    matplotlib.use('Agg')

import numpy as np
import os
import librosa
import h5py
from tqdm import tqdm
import sys
import nsltools as nsl
from scipy.io import loadmat

np.set_printoptions(threshold=sys.maxsize)
tv_dim = 9
tv_timesteps = 200

def min_max_norm(X):
    # X_samples = X.shape[0]
    # X_pitch = X.flatten()
    min_val = np.min(X)
    max_val = np.max(X)
    X_pitch = (X - min_val) / (max_val - min_val)
    #print(std)
    return X_pitch


def generate_data(path_audio, path_params, audio_time=2, sampling_rate=16000, random=False):
    clean_path = path_audio
    directory = path_audio[:-1] + "_XRMB_ext_2sec_new" + ".data"  # save data to this path
    if random:
        directory = path_audio[:-1] + "_24_init" + ".data"  # save data to this path
        random = False

    audio_len = np.int(audio_time * sampling_rate)
    cur_sample = 0  # current sample generated
    file_id = 0

    # collect raw waveform and trim them into equal length, here is 32000 (16k * 2s)
    spk_wav_tmp = np.zeros((10000, audio_len))
    h_tmp = np.zeros((10000, tv_dim, tv_timesteps))
    for (dirpath, dirnames, filenames) in tqdm(os.walk(clean_path)):
        for files in tqdm(filenames):
            if files[-5] != ')' and files[-10] != ')':
                if '.wav' in files:
                    s_wav = clean_path + files
                    s_wav, s_sr = librosa.load(s_wav, sr=sampling_rate)

                    if len(s_wav) < audio_len:
                        spk_wav = np.concatenate([s_wav, np.zeros(audio_len - len(s_wav))])
                    elif len(s_wav) > audio_len:
                        spk_wav = s_wav[:audio_len]

                    spk_wav = np.array(spk_wav).reshape(-1, audio_len)
                    spk_wav_tmp[cur_sample:cur_sample + spk_wav.shape[0]] = spk_wav
                    cur_sample = cur_sample + spk_wav.shape[0]

                # load and process TV files
                tv_file = files[:-4] + '_tv.mat'
                tv_path = path_params + tv_file

                # load TV files
                TV_all_data = loadmat(tv_path)
                TV_data = TV_all_data['tv_norm']

                TV_data = np.transpose(TV_data)

                # Normalized pitch to 0-1 range
                TV_data[:,8] = min_max_norm(TV_data[:,8])

                # TV_data = TV_data.reshape(TV_data.shape[1], TV_data.shape[0])
                tv_len = audio_time* 100 # 100Hz TV sampling rate

                if TV_data.shape[0] < tv_len:
                    pad_amt = tv_len - TV_data.shape[0]
                    TV_data = np.pad(TV_data, ((0, pad_amt), (0, 0)), 'constant', constant_values=(0, 0))
                elif TV_data.shape[0] > tv_len:
                    TV_data = TV_data[0:tv_len, :]

                TV_data = np.transpose(TV_data)
                h_tmp[file_id, :, :] = TV_data
                file_id+=1

    print('trim finished')  # spk_wav_tmp.shape[0]=Number of files in data directory or Num of examples= N

    spk_wav_tmp = spk_wav_tmp[:cur_sample, :]
    h_tmp = h_tmp[:spk_wav_tmp.shape[0], :, :]
    # spk_wav_tmp.shape= (Num of examples (N), 32000)

    spk_tmp = np.zeros((spk_wav_tmp.shape[0], spk_wav_tmp.shape[1]))  # raw speech with normalized power
    # h_tmp = np.zeros((spk_tmp.shape[0], tv_dim, 250))  # ideal hiddens from world
    spec_tmp_513 = np.zeros(
        (spk_wav_tmp.shape[0], 128, 250))  # dimensions of the AudSpec - needs to be softcoded for scalability
    #spec_tmp_513_pw = np.zeros((spk_wav_tmp.shape[0], 128,
    #                           250))  # dimensions of the Reconstructed AudSpec - needs to be softcoded for scalability
    print(spec_tmp_513.shape, 'spec_tmp_513')

    '''Parameters for AudSpectrogram'''
    frmlen = 8
    tc = 8
    paras_c = [frmlen, tc, -2, np.log2(sampling_rate / 16000)]
    print(sampling_rate)

    pad = 40
    for i in tqdm(range(spk_wav_tmp.shape[0])):
        if i % 100 == 0:
            print(i)
        wav = spk_wav_tmp[i, :].copy().astype('float64')
        wav = wav.reshape(-1)
        # wav=nsl.unitseq(wav)    #THis here causes the problem: RuntimeWarning: overflow encountered in exp

        wav = wav / np.sqrt(np.sum(wav ** 2))  # power normalization

        wav = nsl.unitseq(wav)  # THis here causes the problem: RuntimeWarning: overflow encountered in exp

        # #wav.shape=(32000,)
        if not random:
            spk_tmp[i, :] = wav  # this is saved

        spec513 = nsl.wav2aud(wav, paras_c)  # audSpec
        # print (spec513.shape, 'spec513--line 116')
        spec_tmp_513[i, :, 0:250] = spec513.T  # AudSpec
        # print (spec_tmp_513[i,:,:])

    # write data
    dset = h5py.File(directory, 'w')
    print(spk_tmp.shape)
    spk_set = dset.create_dataset('speaker', shape=(spk_tmp.shape[0], spk_tmp.shape[1]), dtype=np.float64)
    hid_set = dset.create_dataset('hidden', shape=(spk_tmp.shape[0], tv_dim, tv_timesteps), dtype=np.float64)

    spec513_set = dset.create_dataset('spec513', shape=(spk_tmp.shape[0], 128, 250), dtype=np.float64)
    # spec_513_pw_set = dset.create_dataset('spec513_pw', shape=(spk_tmp.shape[0], 128, 251), dtype=np.float64)

    spk_set[:, :] = spk_tmp
    hid_set[:, :, :] = h_tmp
    spec513_set[:, :, :] = spec_tmp_513

    dset.close()
    print('finished')


if __name__ == "__main__":
    if (len(sys.argv) == 3 or len(sys.argv) == 4) and sys.argv[1] != "-h":
        if len(sys.argv) == 3:
            generate_data(sys.argv[1], sys.argv[2])
        elif sys.argv[3] == "random":
            generate_data(sys.argv[1], sys.argv[2], random=True)
        else:
            print("We did not understand the second argument.")
    else:
        print("USAGE: python3", sys.argv[0], "<path to the wav data>")


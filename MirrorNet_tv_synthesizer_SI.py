#!/usr/bin/env python
# coding: utf-8

'''
This script implements the MirrorNet model without any initialization step

Author : Yashish

'''

from __future__ import print_function
import argparse
import time

import matplotlib

SERVER = True  # This variable enable or disable matplotlib, set it to true when you use the server!
if SERVER:
    matplotlib.use('Agg')

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim

import os
import numpy as np
import h5py
import time
import subprocess
import logging
from datetime import date
import matplotlib.pyplot as plt
import pickle as pkl
import datetime

from utils_PPMC import compute_corr_score


# In[61]:
spec_len = 250
N = 200  # 2 seconds long with 100Hz sampling rate
params = 9

fs = 16000
# pad_render = 0.1


class BoDataset(Dataset):
    """
        Wrapper for the WSJ Dataset.
    """

    def __init__(self, path):
        super(BoDataset, self).__init__()

        self.h5pyLoader = h5py.File(path, 'r')
        self.wav = self.h5pyLoader['speaker']
        self.spec = self.h5pyLoader['spec513']
        # self.specWorld = self.h5pyLoader['spec513_pw']
        self.hidden = self.h5pyLoader['hidden']
        self._len = self.wav.shape[0]
        # print (path)
        print('h5py loader', type(self.h5pyLoader))
        print('wav', self.wav)
        print('spec', self.spec)
        # print('self.specWorld', self.specWorld)
        print('self.hidden', self.hidden)

    def __getitem__(self, index):
        # print (index)
        wav_tensor = torch.from_numpy(self.wav[index])  # raw waveform with normalized power
        # print (len(self.hidden[index]))
        # spec_tensor = torch.from_numpy(self.spec[index])         # magnitude of spectrogram of raw waveform
        if len(self.hidden[index]) > 0:
            hidden_tensor = torch.from_numpy(self.hidden[index])  # parameters of world
        else:
            hidden_tensor = []
        spec_tensor = torch.from_numpy(self.spec[index])

        return wav_tensor, [], hidden_tensor, spec_tensor

    def __len__(self):
        return self._len


def log_print(content):
    print(content)
    logging.info(content)

def initialize_weights(m):
  if isinstance(m, nn.Conv2d):
      nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
      if m.bias is not None:
          nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight.data, 1)
      nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, nn.Linear):
      nn.init.kaiming_uniform_(m.weight.data)
      nn.init.constant_(m.bias.data, 0)


class synthesis_model(nn.Module):
    def __init__(self):
        super(synthesis_model, self).__init__()

        self.AE_win = 320
        self.AE_stride = 80
        self.AE_channel = 256
        # self.BN_channel = 256
        self.BN_channel = 256
        self.kernel = 3
        self.CNN_layer = 4
        self.stack = 3

        self.sp_seq = nn.Sequential(
            nn.Conv1d(params, 128, 1),
            nn.BatchNorm1d(128, 128, 1),
            #nn.Dropout(p=0.3),
            nn.ReLU(),
            #nn.AvgPool1d(4, stride=4),
            nn.Upsample(scale_factor=5),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256, 256, 1),
            #nn.Dropout(p=0.3),
            nn.ReLU(),
            #nn.Upsample(scale_factor=5),
            nn.AvgPool1d(4, stride=4),
            nn.Conv1d(256, 256, 1),  # Kernel size 1 with 0 padding maintains length of output signal=401
            nn.BatchNorm1d(256, 256, 1),
            #nn.Dropout(p=0.3),
            nn.ReLU()

        )
        self.sp_seq.apply(initialize_weights)

        self.CNN = nn.ModuleList([])           # changed dilation rate order to match encoder
        for s in range(self.stack):
            self.CNN.append(CNN1D(self.BN_channel, self.BN_channel, self.kernel,
                                  dilation=2 ** 0))
            self.CNN.append(CNN1D(self.BN_channel, self.BN_channel, self.kernel,
                                  dilation=2 ** 2))
            self.CNN.append(CNN1D(self.BN_channel, self.BN_channel, self.kernel,
                                  dilation=2 ** 4))

        self.L2 = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.GroupNorm(1, self.AE_channel, eps=1e-16),
            nn.Conv1d(256, 128, 1),
            nn.Conv1d(128, 128, 1)  # added new
            # nn.BatchNorm1d(128, 128, 1),  # added new
            # nn.ReLU()                 # added new

        )

    def forward(self, input):
        sp_out = self.sp_seq(input)

        this_input = sp_out

        for i in range(len(self.CNN)):
            this_input = self.CNN[i](this_input)  # 64-128-250
            # print (this_input.size(), 'Size(this_input) inside CNN stack decoder')

        #this_input = self.final_conv(this_input)

        final_input = self.L2(this_input)
        # print (this_input.size(), 'Size(this_input). Decoder op') #output is 64-128-250

        return final_input


class CNN1D(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel,
                 dilation=1, stride=1, padding=0, ds=2):
        super(CNN1D, self).__init__()

        self.causal = False

        if self.causal:
            self.padding1 = (kernel - 1) * dilation
            self.padding2 = (kernel - 1) * dilation * ds
        else:
            self.padding1 = (kernel - 1) // 2 * dilation
            self.padding2 = (kernel - 1) // 2 * dilation * ds

        self.conv1d1 = nn.Conv1d(input_channel, hidden_channel, kernel,
                                 stride=stride, padding=self.padding1,
                                 dilation=dilation)
        self.conv1d2 = nn.Conv1d(input_channel, hidden_channel, kernel,
                                 stride=stride, padding=self.padding2,
                                 dilation=dilation * ds)

        self.reg1 = nn.BatchNorm1d(hidden_channel, eps=1e-16)
        self.reg2 = nn.BatchNorm1d(hidden_channel, eps=1e-16)
        # self.reg1 = GlobalLayerNorm(hidden_channel)
        # self.reg2 = GlobalLayerNorm(hidden_channel)

        self.nonlinearity = nn.ReLU()

    def forward(self, input):
        # res=input

        output = self.nonlinearity(self.reg1(self.conv1d1(input)))
        if self.causal:
            output = output[:, :, :-self.padding1]
        output = self.reg2(self.conv1d2(output))
        if self.causal:
            output = output[:, :, :-self.padding2]
        output = self.nonlinearity(output + input)
        # output=output+res

        return output


# In[71]:
EPS = 1e-8


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""

    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta

        return gLN_y


class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()

        # self.AE_win = 320
        # self.AE_stride = 80
        self.AE_channel = 256
        self.BN_channel = 256

        self.AE_win = 320
        self.AE_stride = 128
        # self.AE_channel = 128
        # self.BN_channel = 128

        self.kernel = 3
        self.CNN_layer = 4
        self.stack = 3

        self.encoder = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.Conv1d(128, 256, 1),
            nn.GroupNorm(1, 256, eps=1e-16),
            nn.Conv1d(256, 256, 1)  # added new
            # nn.BatchNorm1d(128, 128, 1),  # added new
            # nn.ReLU()
        )

        self.CNN = nn.ModuleList([])
        for s in range(self.stack):
            self.CNN.append(CNN1D(self.BN_channel, self.BN_channel, self.kernel,
                                  dilation=2 ** 0))
            self.CNN.append(CNN1D(self.BN_channel, self.BN_channel, self.kernel,
                                  dilation=2 ** 2))
            self.CNN.append(CNN1D(self.BN_channel, self.BN_channel, self.kernel,
                                  dilation=2 ** 4))

        self.music_seq = nn.Sequential(

            nn.Conv1d(self.BN_channel, 256, 1, bias=False),
            nn.BatchNorm1d(256, 256, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4),
            #nn.AvgPool1d(5, stride=5),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128, 128, 1),
            nn.ReLU(),
            nn.AvgPool1d(5, stride=5),
            nn.Conv1d(128, params, 1, bias=False),
            nn.BatchNorm1d(params, params, 1),
            #nn.AvgPool1d(5, stride=5),
            #nn.Upsample(scale_factor=4),		# uncomment if melody length is 5 notes
            # nn.Sigmoid()
            nn.Tanh()

        )

    def forward(self, input):
        # input shape: B, T
        batch_size = input.size(0)
        nsample = input.size(1)
        # print (input.size(), ' encoder input.size() encoder') #64-32320

        # encoder
        this_input = self.encoder(input)  # B, C, T #T=[{(32320-320)/80}+1]=401
        # print(enc_output.shape)

        # print (this_input.size(), 'this_input after L1, before stack. Encoder') #64-256-401
        for i in range(len(self.CNN)):
            this_input = self.CNN[i](this_input)
        # print (this_input.size(), 'this_input after CNN stack. Ecnoder-------') #64-256-401

        music = self.music_seq(this_input)  # 64-513-401

        return music


# In[72]:


class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()

        self.AE_win = 320
        self.AE_stride = 80
        self.AE_channel = 256
        self.BN_channel = 256
        # self.BN_channel = 128
        self.kernel = 3
        self.CNN_layer = 4
        self.stack = 3

        self.L1 = nn.Sequential(
            nn.Conv1d(params, 128, 1, bias=False),  # changed 1 + 513 * 2 to 16
            nn.BatchNorm1d(128, 128, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=5),
            nn.Conv1d(128, 256, 1, bias=False),  # changed 1 + 513 * 2 to 16
            nn.BatchNorm1d(256, 256, 1),
            nn.ReLU(),
            #nn.Upsample(scale_factor=5),
            nn.AvgPool1d(4, stride=4),
            nn.Conv1d(256, 256, 1, bias=False),  # changed 1 + 513 * 2 to 16
            nn.BatchNorm1d(256, 256, 1),
            nn.ReLU()
        )

        self.CNN = nn.ModuleList([])
        for s in range(self.stack):
            self.CNN.append(CNN1D(self.BN_channel, self.BN_channel, self.kernel,
                                  dilation=2 ** 0))
            self.CNN.append(CNN1D(self.BN_channel, self.BN_channel, self.kernel,
                                  dilation=2 ** 2))
            self.CNN.append(CNN1D(self.BN_channel, self.BN_channel, self.kernel,
                                  dilation=2 ** 4))


        self.L2 = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.GroupNorm(1, self.AE_channel, eps=1e-16),
            nn.Conv1d(256, 128, 1),
            nn.Conv1d(128, 128, 1)  # added new
            # nn.BatchNorm1d(128, 128, 1),  # added new
            # nn.ReLU()    		# added new

        )

    def forward(self, input):
        # input shape: B, T
        batch_size = input.size(0)
        nsample = input.size(1)

        this_input = self.L1(input)  # 64-128-250
        # print(this_input.shape)
        # print (this_input.size(),'this_input before CNN stack decoder--------' )
        for i in range(len(self.CNN)):
            this_input = self.CNN[i](this_input)  # 64-128-250
            # print (this_input.size(), 'Size(this_input) inside CNN stack decoder')
        # print(this_input.shape)
        this_input = self.L2(this_input)
        # print (this_input.size()) #output is 64-128-250

        return this_input


criteron = nn.MSELoss()

def min_max_norm(X):
    # X_samples = X.shape[0]
    # X_mfcc = X.flatten()
    min_val = np.min(X)
    max_val = np.max(X)
    X_norm = (X - min_val) / (max_val - min_val)
    #print(std)
    return X_norm

def new_training_technique(epoch, train_D=False, train_E=False, init=False):
    '''
    This function train the networks for one epoch using the new training technique.
    train_D and train_E can be specified to train only one network or both in the same time.
    More details about the loss functions and the architecture in the README.md
    '''
    start_time = time.time()

    D.eval()
    E.eval()

    if train_D:
        D.train()
    if train_E:
        E.train()

    train_loss1 = 0.
    train_loss2 = 0.
    train_loss = 0.
    new_loss = 0.

    pad = 10

    for (batch_idx, data_random), (_, data_train) in zip(enumerate(train_random), enumerate(train_loader)):

        batch_wav_random = Variable(data_random[0]).contiguous().float()
        batch_h_random = Variable(data_random[2]).contiguous().float()
        batch_spec_random = Variable(data_random[3]).contiguous().float()

        # batch_wav_train = Variable(data_train[0]).contiguous().float()
        # batch_spec_train = Variable(data_train[2]).contiguous().float()

        if args.cuda:
            batch_wav_random = batch_wav_random.cuda()
            batch_spec_random = batch_spec_random.cuda()
            batch_h_random = batch_h_random.cuda()

        h_hat = E(batch_spec_random)
        # h_hat = transform_params(h_hat)
        # print(batch_h_random)
        # h_hat = batch_music #torch.cat([batch_f0, batch_sp, batch_music], 1)

        if train_D and not init:
            # if not init:
            music_tmp = h_hat.data.cpu().numpy().astype('float64')

            for j in range(0,music_tmp.shape[0]):
                for k in range(6,9):
                    music_tmp[j, k, :] = min_max_norm(music_tmp[j, k, :])

            frmlen = 8
            tc = 8
            sampling_rate = fs
            paras_c = [frmlen, tc, -2, np.log2(sampling_rate / 16000.0)]

            music_tensor = Variable(torch.from_numpy(music_tmp)).contiguous().float()

            s2 = TV_syn(music_tensor)

            if args.cuda:
                s2 = s2.cuda()

        if train_D:
            D_optimizer.zero_grad()

            if init:
                loss2 = criteron(D(batch_h_random), batch_spec_random)
                loss2.backward()

            else:
                # loss_D1 = criteron(D(batch_h_random), batch_spec_random)
                loss_D2 = criteron(D(h_hat), s2)
                # print(loss_D2)
                # loss2 = (loss_D1 + loss_D2) / 2
                loss2 = loss_D2
                loss2.backward()

            # print(loss2)
            D_optimizer.step()
            train_loss2 += loss2.item()

        if train_E:
            E_optimizer.zero_grad()

            if init:
                c = E(batch_spec_random)

                loss1 = criteron(c, batch_h_random)

            else:
                loss_E1 = criteron(D(h_hat), batch_spec_random)

                loss1 = loss_E1

            # print(loss1)
            # loss1 = loss1.requires_grad=True
            loss1.backward()
            E_optimizer.step()
            train_loss1 += loss1.item()

        if train_E and train_D:
            loss = loss1 + loss2
            train_loss += loss.item()

        if train_D and not init:
            new_loss += loss_D2.item()

        # loss.backward()
        # all_optimizer.step()

        if (batch_idx + 1) % args.log_step == 0:
            elapsed = time.time() - start_time
            log_print(
                '| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | train loss1 (Encoder) {:5.8f} | train loss2 (Decoder) {:5.8f} | total loss {:5.8f} |'.format(
                    epoch, batch_idx + 1, len(train_loader),
                           elapsed * 1000 / (batch_idx + 1), train_loss1 / (batch_idx + 1),
                           train_loss2 / (batch_idx + 1), train_loss / (batch_idx + 1)))

    train_loss1 /= (batch_idx + 1)
    train_loss2 /= (batch_idx + 1)
    train_loss /= (batch_idx + 1)

    print(len(train_loader), 'len(train_loader)')
    print("Losses calculated")

    log_print('-' * 99)
    log_print(
        '    | end of training epoch {:3d} | time: {:5.2f}s | train loss1 (encoder) {:5.8f} | train loss2 (decoder){:5.8f}|'.format(
            epoch, (time.time() - start_time), train_loss1, train_loss2))

    return train_loss, train_loss1, train_loss2, new_loss


def trainTogether_newTechnique(epochs=None, save_better_model=False, loader_eval="evaluation", train_E=False,
                               train_D=False, init=False, name=""):
    '''
    Glob function for the new training technique, it iterates over all the epochs,
    process the learning rate decay, save the weights and call the evaluation funciton.
    '''
    reduce_epoch = []

    training_loss_encoder = []
    training_loss_decoder = []

    validation_loss_encoder = []
    validation_loss_decoder = []

    new_loss = []

    decay_cnt1 = 0
    decay_cnt2 = 0

    if epochs is None:
        epochs = args.epochs

    print()

    if train_E and not train_D:
        print("TRAINING E ONLY")
    if not train_E and train_D:
        print("TRAINING D ONLY")
    if train_E and train_D:
        print("TRAINING E AND D")
    if not train_E and not train_D:
        print("TRAINING NOTHING ...")

    # Training all
    for epoch in range(1, epochs + 1):
        if args.cuda:
            E.cuda()
            D.cuda()

        error = new_training_technique(epoch, train_D=train_D, train_E=train_E, init=init)

        training_loss_encoder.append(error[1])
        training_loss_decoder.append(error[2])
        new_loss.append(error[3])
        print("Difference between D en W:", error[3])

        decay_cnt2 += 1

        if np.min(training_loss_encoder) not in training_loss_encoder[-3:] and decay_cnt1 >= 3:
            E_scheduler.step()
            decay_cnt1 = 0
            log_print('      Learning rate decreased for E.')
            log_print('-' * 99)

        if np.min(training_loss_decoder) not in training_loss_decoder[-3:] and decay_cnt2 >= 3:
            D_scheduler.step()
            decay_cnt2 = 0
            log_print('      Learning rate decreased for D.')
            log_print('-' * 99)

    del training_loss_encoder
    del training_loss_decoder

def generate_figures(mode="evaluation", name="", load_weights=("", "")):
    '''
    Generates a set of figures that help to evaluate the training, it allows to see the generated ap, f0 and sp for example.
    '''

    if mode == "evaluation":
        loader = validation_loader
    elif mode == "train":
        loader = train_loader
    elif mode == "train_random":
        loader = train_random
    else:
        print("We did not understand the mode, we skip this function.")
        return

    if name is not "":
        local_out_init = out + "/" + name + "_" + mode
    else:
        local_out_init = out + "/" + mode

    if not os.path.exists(local_out_init):
        os.makedirs(local_out_init)

    if load_weights != ("", ""):
        E.load_state_dict(torch.load(load_weights[0]))
        D.load_state_dict(torch.load(load_weights[1]))

    E.eval()
    D.eval()

    if args.cuda:
        E.cuda()
        D.cuda()

    pad = 10

    for batch_idx, data in enumerate(loader):

        batch_wav = Variable(data[0]).contiguous().float()
        batch_h = Variable(data[2]).contiguous().float()
        batch_spec = Variable(data[3]).contiguous().float()

        if args.cuda:
            batch_wav = batch_wav.cuda()
            batch_spec = batch_spec.cuda()
            batch_h = batch_h.cuda()


        # predict parameters through waveform
        batch_h_hat = E(batch_spec)

        music_tmp = batch_h_hat[:, 0:params, :].data.cpu().numpy().astype('float64')
        music = batch_h[:, 0:params, :].data.cpu().numpy().astype('float64')

        for j in range(0, music_tmp.shape[0]):
            for k in range(6, 9):
                music_tmp[j, k, :] = min_max_norm(music_tmp[j, k, :])

        if not os.path.exists(local_out_init + "/waveforms"):
            os.makedirs(local_out_init + "/waveforms")

        frmlen = 8
        tc = 8
        sampling_rate = fs
        paras_c = [frmlen, tc, -2, np.log2(sampling_rate / 16000.0)]

        music_tensor = Variable(torch.from_numpy(music_tmp)).contiguous().float()

        spec_wav = TV_syn(music_tensor)

        #music = batch_h[:, 0:params, :].data.cpu().numpy().astype('float64')
        ## SAVE ALL FIGURES
        wave_files = batch_wav[:, :].data.cpu().numpy().astype('float64')


        # batch_h_random = torch.cat([f0_random, sp_random, ap_random], 1)

        local_out_init += "/lentent_space/"
#
        if mode == "evaluation":
            corr_avg, avg_corr, avg_corr_6TVs = compute_corr_score(music_tmp, music, params)

            # write a .txt with results and model params
            lines = ['Corr_Average:' + str(corr_avg), 'Corr_Average_all:' + str(avg_corr), 'Corr_Average_6TVs:' + str(avg_corr_6TVs)]
            with open(out_dir + 'log.txt', 'w') as f:
                f.write('\n'.join(lines))


        # local_out_init += "/lentent_space/"
        # print (f0.shape[0])
        num = np.min((music_tmp.shape[0], 10))
        # print (num, 'num')
        # Save ap
        # Save Sp
        music_hat_saved = np.zeros((num, params, N), dtype=np.float64)
        wave_files_saved = np.zeros((num, wave_files.shape[1]), dtype=np.float64)
        # sp_hat_saved = np.zeros((num, 513, 251), dtype=np.float64)
        music_ideal_saved = np.zeros((num, params, N), dtype=np.float64)


        for i in range(num):
            music_hat_saved[i, :, :] = music_tmp[i, :, :]
            wave_files_saved[i, :] = wave_files[i, :]
            music_ideal_saved[i, :, :] = music[i, :, :]

        if not os.path.exists(local_out_init):
            os.makedirs(local_out_init)

        with open(local_out_init + 'ap_hat.pkl', 'wb') as f:
            pkl.dump(music_hat_saved, f)

        with open(local_out_init + 'wave_files.pkl', 'wb') as f:
            pkl.dump(wave_files_saved, f)

        with open(local_out_init + 'ap_ideal.pkl', 'wb') as f:
            pkl.dump(music_ideal_saved, f)

        if not os.path.exists(local_out_init):
            os.makedirs(local_out_init)

        for i in range(min(music_hat_saved.shape[0], 10)):

            local_out = local_out_init + str(i) + "/"

            if not os.path.exists(local_out):
                os.makedirs(local_out)

            # params_labels_all = ['LA','LP', 'TBCL', 'TBCD', 'TTCL', 'TTCD', 'Aperiodicty', 'Periodicity', 'Pitch']
            # params_labels = ['Tongue Diameter', 'Voiceness', 'Nasalance', 'Lips', 'pitch']

            plt.title("LA")
            plt.subplot(121)
            plt.title("from encoder (E(x))")
            plt.plot(music_tmp[i, 0, :])
            plt.subplot(122)
            plt.title("ground truth")
            plt.plot(music[i, 0, :])
            plt.savefig(local_out + "LA.eps")

            if not SERVER:
                plt.show()
            else:
                plt.close()

            plt.title("LP")
            plt.subplot(121)
            plt.title("from encoder (E(x))")
            plt.plot(music_tmp[i, 1, :])
            plt.subplot(122)
            plt.title("ground truth")
            plt.plot(music[i, 1, :])
            plt.savefig(local_out + "LP.eps")

            if not SERVER:
                plt.show()
            else:
                plt.close()

            plt.title("TBCL")
            plt.subplot(121)
            plt.title("from encoder (E(x))")
            plt.plot(music_tmp[i, 2, :])
            plt.subplot(122)
            plt.title("ground truth")
            plt.plot(music[i, 2, :])
            plt.savefig(local_out + "TBCL.eps")

            if not SERVER:
                plt.show()
            else:
                plt.close()

            plt.title("TBCD")
            plt.subplot(121)
            plt.title("from encoder (E(x))")
            plt.plot(music_tmp[i, 3, :])
            plt.subplot(122)
            plt.title("ground truth")
            plt.plot(music[i, 3, :])
            plt.savefig(local_out + "TBCD.eps")

            if not SERVER:
                plt.show()
            else:
                plt.close()

            plt.title("TTCL")
            plt.subplot(121)
            plt.title("from encoder (E(x))")
            plt.plot(music_tmp[i, 4, :])
            plt.subplot(122)
            plt.title("ground truth")
            plt.plot(music[i, 4, :])
            plt.savefig(local_out + "TTCL.eps")

            if not SERVER:
                plt.show()
            else:
                plt.close()

            plt.title("TTCD")
            plt.subplot(121)
            plt.title("from encoder (E(x))")
            plt.plot(music_tmp[i, 5, :])
            plt.subplot(122)
            plt.title("ground truth")
            plt.plot(music[i, 5, :])
            plt.savefig(local_out + "TTCD.eps")

            if not SERVER:
                plt.show()
            else:
                plt.close()

            plt.title("Aperiodicity")
            plt.subplot(121)
            plt.title("from encoder (E(x))")
            plt.plot(music_tmp[i, 6, :])
            plt.subplot(122)
            plt.title("ground truth")
            plt.plot(music[i, 6, :])
            plt.savefig(local_out + "Aperio.eps")

            if not SERVER:
                plt.show()
            else:
                plt.close()

            plt.title("Periodicity")
            plt.subplot(121)
            plt.title("from encoder (E(x))")
            plt.plot(music_tmp[i, 7, :])
            plt.subplot(122)
            plt.title("ground truth")
            plt.plot(music[i, 7, :])
            plt.savefig(local_out + "Perio.eps")

            if not SERVER:
                plt.show()
            else:
                plt.close()

            plt.title("Pitch (scaled)")
            plt.subplot(121)
            plt.title("from encoder (E(x))")
            plt.plot(music_tmp[i, 8, :])
            plt.subplot(122)
            plt.title("ground truth")
            plt.plot(music[i, 8, :])
            plt.savefig(local_out + "pitch.eps")

            if not SERVER:
                plt.show()
            else:
                plt.close()

            plt.title("TV synthesizer investigation")
            ax1 = plt.subplot(211)
            plt.title("TV_syn(h_hat)")
            plt.imshow(spec_wav.detach().cpu().numpy()[i, :, :], cmap='jet', aspect="auto", origin='lower')
            plt.colorbar()

            ax2 = plt.subplot(212, sharex=ax1)
            plt.title("TV_hat")
            plt.imshow(music_tmp[i, :, :], cmap=plt.cm.BuPu_r, aspect="auto")
            plt.colorbar()
            plt.savefig(local_out + "/groundtruth_TV_VS_estimated_TV.eps")

            if not SERVER:
                plt.show()
            else:
                plt.close()

            plt.title("TV investigation")
            ax1 = plt.subplot(211)
            plt.title("D(h_hat)")
            plt.imshow(D(batch_h_hat).detach().cpu().numpy()[i, :, :], cmap='jet', aspect="auto", origin='lower')
            plt.colorbar()

            ax2 = plt.subplot(212, sharex=ax1)
            plt.title("TV_hat")
            plt.imshow(music_tmp[i, :, :], cmap=plt.cm.BuPu_r, aspect="auto")
            plt.colorbar()
            plt.savefig(local_out + "/D(h_hat)_VS_TV_hat.eps")

            if not SERVER:
                plt.show()
            else:
                plt.close()

            plt.title("Spectogram")
            plt.subplot(121)
            plt.imshow(batch_spec[i, :, :].cpu().numpy(), cmap='jet', origin='lower')
            plt.title("original")
            plt.subplot(122)
            plt.imshow(D(batch_h_hat).detach().cpu().numpy()[i, :, :], cmap='jet', origin='lower')
            plt.title("generated")
            plt.colorbar()
            plt.savefig(local_out + "/spectrogram.eps")

            if not SERVER:
                plt.show()
            else:
                plt.close()

        return  # Do only the first batch


def getSpectrograms(mode="evaluation"):
    '''
    Generates spectrogram for the original sound (x), D(ideal_h), D(E(x)), world(E(x)) in order to evaluate
    the accuracy and the decoder and the encoder separatly.
    '''
    if mode == "train":
        loader = train_loader

    elif mode == "evaluation":
        loader = validation_loader

    elif mode == "train_random":
        loader = train_random
    else:
        print("The mode is not understood, therefore we don't compute spectrograms.")
        return [([], [], [], [])]

    E.eval()
    D.eval()

    if args.cuda:
        E.cuda()
        D.cuda()

    pad = 10

    frmlen = 8
    tc = 8
    sampling_rate = fs
    paras_c = [frmlen, tc, -2, np.log2(sampling_rate / 16000.0)]

    spectrograms = []

    for batch_idx, data in enumerate(loader):

        batch_wav = Variable(data[0]).contiguous().float()
        batch_h = Variable(data[2]).contiguous().float()
        batch_spec = Variable(data[3]).contiguous().float()

        if args.cuda:
            batch_wav = batch_wav.cuda()
            batch_spec = batch_spec.cuda()
            batch_h = batch_h.cuda()

        # predict parameters through waveform
        batch_h_hat = E(batch_spec)
        # fs = 8000
        music_tmp = batch_h_hat[:, 0:params, :].data.cpu().numpy().astype('float64')

        for j in range(0, music_tmp.shape[0]):
            for k in range(6, 9):
                music_tmp[j, k, :] = min_max_norm(music_tmp[j, k, :])

        music_tensor = Variable(torch.from_numpy(music_tmp)).contiguous().float()

        spec_wav = TV_syn(music_tensor)

        #####################################################################################################################

        realSpectrogram = np.array(batch_spec.detach().cpu().numpy())
        decoderSpectrogram = np.array(D(batch_h).detach().cpu().numpy())
        modelSpectrogram = np.array(D(batch_h_hat).detach().cpu().numpy())
        worldSpectrogram = np.array(spec_wav.detach().cpu().numpy())

        spectrograms.append((realSpectrogram, decoderSpectrogram, modelSpectrogram, worldSpectrogram))
        # In order to only save a few spectrograms (64)
        return spectrograms

    return spectrograms


def plotSpectrograms(spectrograms, name="", MAX_examples=10):
    '''
    Plot or save the spectrograms.
    '''

    if not os.path.exists(out + "/spectrograms/" + name):
        os.makedirs(out + "/spectrograms/" + name)

    if name == 'TRAIN_evaluation_data':
        realSpectrogram, decoderSpectrogram, modelSpectrogram, worldSpectrogram = spectrograms[0]
        # # with open('Pitch_1.pkl', 'wb') as f:
        # #     pkl.dump(X_pitch, f)
        for i in range(min(len(spectrograms[0][0]), MAX_examples)):
            with open(out + "/spectrograms/" + name + "/RealSpectrogram%d_eval.pkl" % (i), 'wb') as f:
                pkl.dump(realSpectrogram[i], f)

            with open(out + "/spectrograms/" + name +"/decoderSpectrogram%d.pkl" % (i), 'wb') as f:
               pkl.dump(decoderSpectrogram[i], f)

            with open(out + "/spectrograms/" + name + "/modelSpectrogram%d_eval.pkl" % (i), 'wb') as f:
                pkl.dump(modelSpectrogram[i], f)

            with open(out + "/spectrograms/" + name +"/VOCSpectrogram%d_eval.pkl" % (i), 'wb') as f:
                pkl.dump(worldSpectrogram[i], f)

    if name == 'TRAIN_train_data':
        realSpectrogram, decoderSpectrogram, modelSpectrogram, worldSpectrogram = spectrograms[0]
        # # with open('Pitch_1.pkl', 'wb') as f:
        # #     pkl.dump(X_pitch, f)
        for i in range(min(len(spectrograms[0][0]), MAX_examples)):
            with open(out + "/spectrograms/" + name +"/RealSpectrogram%d.pkl" % (i), 'wb') as f:
                pkl.dump(realSpectrogram[i], f)

            with open(out + "/spectrograms/" + name +"/decoderSpectrogram%d.pkl" % (i), 'wb') as f:
               pkl.dump(decoderSpectrogram[i], f)

            with open(out + "/spectrograms/" + name +"/modelSpectrogram%d.pkl" % (i), 'wb') as f:
                pkl.dump(modelSpectrogram[i], f)

            with open(out + "/spectrograms/" + name +"/VOCSpectrogram%d.pkl" % (i), 'wb') as f:
                pkl.dump(worldSpectrogram[i], f)


    for i in range(min(len(spectrograms[0][0]), MAX_examples)):

        realSpectrogram, decoderSpectrogram, modelSpectrogram, worldSpectrogram = spectrograms[0]

        plt.subplot(221)
        plt.imshow(realSpectrogram[i], cmap='jet', origin='lower')  # Ground Truth
        plt.title('Real Spectrogram')
        # plt.savefig(out + "/spectrograms/" + name + "/original_spec_" + str(i) + ".eps")

        plt.subplot(222)
        plt.imshow(decoderSpectrogram[i], cmap='jet', origin='lower')
        plt.title('Decoder Spectrogram (using ideal h)')  # D(ideal_h)

        plt.subplot(223)
        plt.imshow(modelSpectrogram[i], cmap='jet', origin='lower')
        if name == 'TRAIN_evaluation_data':
            np.save("spectrogram_{}".format(i), modelSpectrogram[i])
        plt.title('Model Spectrogram: D(E(x))')  # (D(E(X)))
        # plt.savefig(out + "/spectrograms/" + name + "/model_spec_" + str(i) + ".eps")

        plt.subplot(224)
        plt.imshow(worldSpectrogram[i], cmap='jet', origin='lower')
        plt.title('Spectrogram from TV Syn (from encoder h)')  # World(E(H))
        plt.savefig(out + "/spectrograms/" + name + "/out_specs_" + str(i) + ".eps")

        #plt.savefig(out + "/spectrograms/" + name + "/" + str(i) + ".eps")

        # plot DIVA spectrogram separately
        # plt.imshow(worldSpectrogram[i], cmap='jet', origin='lower', aspect="auto")
        # plt.title('Spectrogram from DIVA (from encoder h)')  # World(E(H))
        # plt.savefig(out + "/spectrograms/" + name +"/diva_spectrogram_" + str(i) + ".eps")

        if not SERVER:
            plt.show()
        else:
            plt.close()


# prepare exp folder
exp_name = 'MirrorNet'
descripiton = 'train MirrorNet without ideal hiddens'
exp_new = 'tmp/'
base_dir = './'

### Set the name of the folder
out = "figs/NEW_with_only_1_loss_for_SI_noinit_" + str(date.today()) + "_H_" + str(datetime.datetime.now().hour)
out_dir = "./figs/NEW_with_only_1_loss_for_SI_noinit_" + str(date.today()) + "_H_" + str(datetime.datetime.now().hour) + "/"

#model_dir = "./figs/NEW_with_only_1_loss_for_speech_DATE_SI_9TVs_2022-04-14_H_4/"
model_dir = "./figs/NEW_with_only_1_loss_for_speech_DATE_syn_9TVs_2022-10-05_H_1/"

# Create sub directory is dont exist
if not os.path.exists(out_dir + exp_new):
    os.makedirs(out_dir + exp_new)

# Create weights directory if don't exist
if not os.path.exists(out_dir + exp_new + 'net/'):
    os.makedirs(out_dir + exp_new + 'net/')


if not os.path.exists(base_dir + exp_new):
    os.makedirs(base_dir + exp_new)

# Create log directory if don't exist
if not os.path.exists(base_dir + exp_new + "log/"):
    os.makedirs(base_dir + exp_new + "log/")


# setting logger   NOTICE! it will overwrite current log
log_dir = base_dir + exp_new + "log/" + str(date.today()) + ".log"
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    filename=log_dir,
                    filemode='a')

# global params

parser = argparse.ArgumentParser(description=exp_name)
parser.add_argument('--batch-size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--decoder_lr', type=float, default=1e-3,  # changed from 3e-4
                    help='learning rate')
parser.add_argument('--encoder_lr', type=float, default=1e-3,  # changed from 3e-4
                    help='learning rate')
parser.add_argument('--seed', type=int, default=20190101,
                    help='random seed')
parser.add_argument('--val-save', type=str, default=base_dir + exp_new + '/' + exp_name + '/net/cv/model_weight.pt',
                    help='path to save the best model')

parser.add_argument('--train_data', type=str,
                    default=base_dir + "SI_data/New_train_files/train_audio_XRMB_ext_2sec_new.data",
                    help='path to training data')

parser.add_argument('--test_data', type=str,
                    default=base_dir + "SI_data/New_train_files/test_audio_XRMB_ext_2sec_new.data",
                    help='path to testing data')

# Not used
parser.add_argument('--initialization_data', type=str,
                    default=base_dir + "SI_data/New_train_files/dev_audio_XRMB_ext_2sec_new.data",
                    help='path to initialization data')

parser.add_argument('--train_random_data', type=str,
                    default=base_dir + "SI_data/dev_audio_XRMB_ext_2sec.data",
                    help='path to train random data')

args, _ = parser.parse_known_args()
print(type(args))
print(args)
np.save(base_dir + exp_new + '/model_arch', args)
args.cuda = args.cuda and torch.cuda.is_available()

np.random.seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True}
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA ACTIVATED")
else:
    kwargs = {}
    print("CUDA DISABLED")

training_data_path = args.train_data
validation_data_path = args.test_data
initialization_data_path = args.initialization_data
train_random_data_path = args.train_random_data

# initialization_loader = DataLoader(BoDataset(initialization_data_path),
#                           batch_size=args.batch_size,
#                           shuffle=False,
#                           **kwargs)

train_loader = DataLoader(BoDataset(training_data_path),
                          batch_size=args.batch_size,
                          shuffle=False,
                          **kwargs)
# print ('----------------------', list(train_loader))
# print ('.,.,.,.,.,.,.,.,.,.', (list(train_loader)[0]))

train_random = DataLoader(BoDataset(train_random_data_path),
                          batch_size=args.batch_size,
                          shuffle=False,
                          **kwargs)

validation_loader = DataLoader(BoDataset(validation_data_path),
                               batch_size=args.batch_size,
                               shuffle=False,
                               **kwargs)

eval_byPieces_loader = DataLoader(BoDataset(validation_data_path),
                                  batch_size=1,
                                  shuffle=False,
                                  **kwargs)

args.dataset_len = len(train_loader)
args.log_step = args.dataset_len // 2

E = encoder()
D = decoder()
# print ('. . . . . . . . . . ', list(E.parameters()))

if args.cuda:
    E.cuda()
    D.cuda()

TV_syn = synthesis_model()

checkpoint_D = torch.load(model_dir + exp_new + 'net/synthesizer_1.pt')
TV_syn.load_state_dict(checkpoint_D['model_state_dict'])
# D_optimizer.load_state_dict(checkpoint_D['optimizer_state_dict'])

# E.load_state_dict(torch.load('tmp/net/Gui_model_weight_E1.pt'))
# D.load_state_dict(torch.load('tmp/net/Gui_model_weight_D1.pt'))


current_lr = args.lr
E_optimizer = optim.Adam(E.parameters(), lr=args.encoder_lr)
E_scheduler = torch.optim.lr_scheduler.ExponentialLR(E_optimizer, gamma=0.5)  # changed gamma=0.5 to 0.3
# E_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(E_optimizer, mode='min', factor=0.5, patience=20, verbose=True, threshold=1e-7)

D_optimizer = optim.Adam(D.parameters(), lr=args.decoder_lr)
D_scheduler = torch.optim.lr_scheduler.ExponentialLR(D_optimizer, gamma=0.5)  # changed gamma=0.5 to 0.1
# D_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(D_optimizer, mode='min', factor=0.2, patience=20, verbose=True, threshold=1e-7)

parameters = [p for p in E.parameters()] + [p for p in D.parameters()]
all_optimizer = optim.Adam(parameters, lr=args.lr)
all_scheduler = torch.optim.lr_scheduler.ExponentialLR(all_optimizer, gamma=0.5)  # changed gamma=0.5 to 0.2

# output info

log_print('Experiment start time is {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
log_print('-' * 70)
log_print('Discription:'.format(descripiton))
log_print('-' * 70)
log_print(args)

log_print('-' * 70)
s = 0
for param in E.parameters():
    s += np.product(param.size())
for param in D.parameters():
    s += np.product(param.size())

log_print('# of parameters: ' + str(s))

log_print('-' * 70)
log_print('Training Set is {}'.format(training_data_path))
log_print('CV set is {}'.format(validation_data_path))
log_print('-' * 70)

### Set the name of the folder
out = "figs/NEW_with_only_1_loss_for_SI_noinit_" + str(date.today()) + "_H_" + str(datetime.datetime.now().hour)

if not os.path.exists(out):
    os.makedirs(out)

if not os.path.exists(out + "/loss"):
    os.makedirs(out + "/loss")

print(out)

# generated the spectrograms before anything to make sure there is no issue with already trained weights or something.
spec = getSpectrograms("train")
plotSpectrograms(spec, "before_training_train_data")

# '''
#         INITIALIZATION
# '''
# print("INITIALIZATION")
#
# # trainTogether_newTechnique(epochs=10, init=True, train_D=True, train_E=True, loader_eval="train", save_better_model=True)
# trainTogether_newTechnique(epochs=50, init=True, train_D=True, train_E=True, loader_eval="train",
#                            save_better_model=True)
#
# print("************************************************************************************************************")
# print("Initialization done")
# print("************************************************************************************************************")
#
# generate_figures("train_random", name="init")  ## evaluation or train
# generate_figures("train", name="init")  ## evaluation or train
#
# spec = getSpectrograms("train_random")
# plotSpectrograms(spec, "INIT_train_random_data")
#
# # spec = getSpectrograms("train")
# # plotSpectrograms(spec, "INIT_train_data")
#
# generate_figures("evaluation", name="init")  ## evaluation or train
# spec = getSpectrograms("evaluation")
# plotSpectrograms(spec, "INIT_evaluation_data")

'''
        TRAINING
'''
print("TRAINING")
for i in range(150):
    #torch.cuda.empty_cache()
    print("ITERATION", str(i + 1))
    trainTogether_newTechnique(epochs=15, name="train_D_" + str(i + 1), init=False, train_D=True, train_E=False,
                               loader_eval="train", save_better_model=False)  # 20 to 5
    trainTogether_newTechnique(epochs=10, name="train_E_" + str(i + 1), init=False, train_D=False, train_E=True,
                               loader_eval="train", save_better_model=False)
    # trainTogether_newTechnique(epochs=10, name="train_D_" + str(i + 1), init=False, train_D=True, train_E=False,
    #                          loader_eval="train", save_better_model=False)  # 20 to 5

    if i % 1000 == 0:
        generate_figures("train", name="still_training_train" + str(i))
        generate_figures("evaluation", name="still_training_eval" + str(i))

print("************************************************************************************************************")
print("Training done")
print("************************************************************************************************************")

generate_figures("train", name="end")  ## evaluation or train
generate_figures("evaluation", name="end")  ## evaluation or train

spec = getSpectrograms("train")
plotSpectrograms(spec, "TRAIN_train_data")

spec = getSpectrograms("evaluation")
plotSpectrograms(spec, "TRAIN_evaluation_data")

# save model with checkpoint
with open(out_dir + exp_new + '/net/voc_model_weight_E1.pt', 'wb') as f:
    torch.save({'model_state_dict': E.cpu().state_dict(), 'optimizer_state_dict': E_optimizer.state_dict()}, f)
    print("We saved encoder!")

with open(out_dir + exp_new + '/net/voc_model_weight_D1.pt', 'wb') as f:
    torch.save({'model_state_dict': D.cpu().state_dict(), 'optimizer_state_dict': D_optimizer.state_dict()}, f)
    print("We saved decoder!")

print("Done.")

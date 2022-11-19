#!/usr/bin/env python
# coding: utf-8

'''
This script implements the articulatory synthesizer which uses the entire train set and source features as inputs

Author : Yashish

'''

from __future__ import print_function
import argparse

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
import logging
from datetime import date
import matplotlib.pyplot as plt
import pickle as pkl
import datetime

# In[61]:
tv_dim = 9 #32 # 64
# ap_enc_dim = 1

fs = 16000

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


# In[70]:
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

        # self.conv1x1 = nn.Conv1d(input_channel, hidden_channel, 1, bias=False)

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
        # self.nonlinearity = nn.PReLU()

    def forward(self, input):
        # res=input

        # input_1=self.nonlinearity(self.reg1(self.conv1x1(input)))
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
            nn.Conv1d(tv_dim, 128, 1),
            nn.BatchNorm1d(128, 128, 1),
            #nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Upsample(scale_factor=5),
            #nn.AvgPool1d(4, stride=4),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256, 256, 1),
            #nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.AvgPool1d(4, stride=4),
            #nn.Upsample(scale_factor=5),
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

        # # this_input = torch.cat([f0_out, sp_out, ap_out], 1)
        this_input = sp_out

        # print (this_input.size(),'this_input before CNN stack decoder--------' )
        for i in range(len(self.CNN)):
            this_input = self.CNN[i](this_input)  # 64-128-250

        final_input = self.L2(this_input)
        # print (this_input.size(), 'Size(this_input). Decoder op') #output is 64-128-250

        return final_input


# criteron = nn.MSELoss()
criteron = nn.L1Loss()


def new_training_technique(epoch, init=False):
    '''
    This function train the networks for one epoch using the new training technique.
    train_D and train_E can be specified to train only one network or both in the same time.
    More details about the loss functions and the architecture in the README.md
    '''
    start_time = time.time()

    model.eval()

    model.train()

    train_loss1 = 0.
    train_loss = 0.
    train_loss = 0.
    new_loss = 0.
    E2_train = 0.  # Decoder Error ||W(H_hat), D(H_Hat)||
    # E1_train = 0.  # Encoder Error || D(E(X)), X ||

    pad = 40

    if init:
        for (batch_idx, data_random) in enumerate(train_loader):

            batch_wav_random = Variable(data_random[0]).contiguous().float()
            batch_h_random = Variable(data_random[2]).contiguous().float()
            batch_spec_random = Variable(data_random[3]).contiguous().float()

            if args.cuda:
                batch_wav_random = batch_wav_random.cuda()
                batch_spec_random = batch_spec_random.cuda()
                batch_h_random = batch_h_random.cuda()

            # print(batch_h_random[0].shape)

            fs = 16000

            model_optimizer.zero_grad()

            if init:
                loss = criteron(model(batch_h_random[:, :, :]), batch_spec_random)
                loss.backward()

            model_optimizer.step()
            train_loss += loss.data.item()  # adding the loss for each batch (for init and for train)

            if (batch_idx + 1) % args.log_step == 0:
                elapsed = time.time() - start_time
                log_print(
                    '| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | train loss: {:5.8f}'.format(
                        epoch, batch_idx + 1, len(train_loader),
                               elapsed * 1000 / (batch_idx + 1), train_loss / (batch_idx + 1)))

        val_loss = 0.0
        model.eval()  # Optional when not using Model Specific layer

        for (batch_idx_val, data_random_val) in enumerate(dev_loader):
            # Transfer Data to GPU if available
            batch_wav_random = Variable(data_random_val[0]).contiguous().float()
            batch_h_random = Variable(data_random_val[2]).contiguous().float()
            batch_spec_random = Variable(data_random_val[3]).contiguous().float()

            if args.cuda:
                batch_wav_random = batch_wav_random.cuda()
                batch_spec_random = batch_spec_random.cuda()
                batch_h_random = batch_h_random.cuda()

            loss = criteron(model(batch_h_random[:, :, :]), batch_spec_random)
            # Calculate Loss
            val_loss += loss.item()

    # train_loss1 /= (batch_idx + 1)
    train_loss /= (batch_idx + 1)  # Decoder error per epoch.
    # train_loss /= (batch_idx + 1)  # total loss
    new_loss /= (batch_idx + 1)
    val_loss /= (batch_idx_val + 1)

    log_print('-' * 99)
    log_print(
        '    | end of training epoch {:3d} | time: {:5.2f}s | train loss: {:5.8f}| val loss: {:5.8f}'.format(
            epoch, (time.time() - start_time), train_loss, val_loss))

    return train_loss, new_loss


def trainTogether_newTechnique(epochs=None, save_better_model=False, loader_eval="evaluation", init=False, name=""):
    '''
    Glob function for the new training technique, it iterates over all the epochs,
    process the learning rate decay, save the weights and call the evaluation funciton.
    '''
    reduce_epoch = []

    training_loss_encoder = []
    training_loss = []

    validation_loss_encoder = []
    validation_loss_decoder = []

    validation_temp_encoder = []
    validation_temp_decoder = []

    # E2_loss_decoder=[]
    E1_loss_encoder = []

    new_loss = []

    decay_cnt1 = 0
    decay_cnt2 = 0

    if epochs is None:
        epochs = args.epochs

    print()

    # Training all
    for epoch in range(1, epochs + 1):
        print(epoch, 'epoch')
        if args.cuda:
            # E.cuda()
            model.cuda()

        error = new_training_technique(epoch, init=init)

        # training_loss_encoder.append(error[1])
        training_loss.append(error[0])
        new_loss.append(error[1])  # E2 error for decoder during training

        decay_cnt2 += 1

        if np.min(training_loss) not in training_loss[-3:] and decay_cnt2 >= 3:
            model_scheduler.step()
            decay_cnt2 = 0
            log_print('      Learning rate decreased for D.')
            log_print('-' * 99)


def getSpectrograms(mode="evaluation"):
    '''
    Generates spectrogram outputs.
    '''
    if mode == "train":
        loader = train_loader

    elif mode == "evaluation":
        loader = validation_loader

    elif mode == "train_random":
        #loader = train_random
        loader = initialization_loader

    else:
        print("The mode is not understood, therefore we don't compute spectrograms.")
        return [([], [], [], [])]

    # E.eval()
    model.eval()

    if args.cuda:
        # E.cuda()
        model.cuda()

    pad = 40

    frmlen = 8
    tc = 8
    sampling_rate = 16000
    paras_c = [frmlen, tc, -2, np.log2(sampling_rate / 16000)]

    spectrograms = []

    for batch_idx, data in enumerate(loader):

        batch_wav = Variable(data[0]).contiguous().float()
        batch_h = Variable(data[2]).contiguous().float()
        batch_spec = Variable(data[3]).contiguous().float()

        if args.cuda:
            batch_wav = batch_wav.cuda()
            batch_spec = batch_spec.cuda()
            batch_h = batch_h.cuda()

        fs = 16000
        spec_wav = np.zeros((batch_wav.shape[0], 128, 250)).astype('float64')

        #####################################################################################################################

        realSpectrogram = np.array(batch_spec.detach().cpu().numpy())
        decoderSpectrogram = np.array(model(batch_h).detach().cpu().numpy())

        spectrograms.append((realSpectrogram, decoderSpectrogram))
        # In order to only save a few spectrograms (64)
        return spectrograms

    return spectrograms


def plotSpectrograms(spectrograms, name="", MAX_examples=10):
    '''
    Plot or save the spectrograms.
    '''

    if not os.path.exists(out + "/spectrograms/" + name):
        os.makedirs(out + "/spectrograms/" + name)

    if name == "TRAIN_evaluation_data" or name == "TRAIN_train_data":
        realSpectrogram, decoderSpectrogram = spectrograms[0]
        # # with open('Pitch_1.pkl', 'wb') as f:
        # #     pkl.dump(X_pitch, f)
        for i in range(min(len(spectrograms[0][0]), MAX_examples)):
            with open(out + "/spectrograms/" + name + "/" + "RealSpectrogram%d.pkl" % (i), 'wb') as f:
                pkl.dump(realSpectrogram[i], f)

            with open(out+"/spectrograms/"+ name + "/" + "decoderSpectrogram%d.pkl" %(i), 'wb') as f:
               pkl.dump(decoderSpectrogram[i], f)

            # with open(out + "/spectrograms/" + "modelSpectrogram%d.pkl" % (i), 'wb') as f:
            #     pkl.dump(modelSpectrogram[i], f)


    for i in range(min(len(spectrograms[0][0]), MAX_examples)):

        realSpectrogram, decoderSpectrogram = spectrograms[0]

        plt.subplot(211)
        plt.imshow(realSpectrogram[i], cmap='jet', origin='lower')  # Ground Truth
        plt.title('Real Spectrogram contrasted')

        plt.subplot(212)
        plt.imshow(decoderSpectrogram[i], cmap='jet', origin='lower')
        plt.title('Decoder Spectrogram (using ideal h)')  # D(ideal_h)

        plt.savefig(out + "/spectrograms/" + name + "/" + str(i) + ".eps")

        if not SERVER:
            plt.show()
        else:
            plt.close()


# prepare exp folder
exp_name = 'Speech_Synthesis'
descripiton = 'train Speech Synthesis from TVs'
exp_new = 'tmp/'
base_dir = './'

### Set the name of the folder
out = "figs/NEW_with_only_1_loss_for_speech_DATE_syn_9TVs_" + str(date.today()) + "_H_" + str(datetime.datetime.now().hour)
out_dir = "./figs/NEW_with_only_1_loss_for_speech_DATE_syn_9TVs_" + str(date.today()) + "_H_" + str(datetime.datetime.now().hour) + "/"

model_dir = "./figs/NEW_with_only_1_loss_for_E_DATE_2021-06-02_H_1/"

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

# Create weights directory if don't exist
#if not os.path.exists(base_dir + exp_new + '/net/'):
#   os.makedirs(base_dir + exp_new + '/net/')

# setting logger   NOTICE! it will overwrite current log
log_dir = base_dir + exp_new + "log/" + str(date.today()) + ".log"
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    filename=log_dir,
                    filemode='a')

# global params

parser = argparse.ArgumentParser(description=exp_name)
parser.add_argument('--batch-size', type=int, default=16,      # changed the batch size from 64
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--decoder_lr', type=float, default=1e-3, # changed from 3e-4
                    help='learning rate')
parser.add_argument('--encoder_lr', type=float, default=1e-3, # changed from 3e-4
                    help='learning rate')
parser.add_argument('--lr', type=float, default=1e-3,    # changed from 1e-3
                    help='learning rate')
parser.add_argument('--seed', type=int, default=20190101,
                    help='random seed')
parser.add_argument('--val-save', type=str, default=base_dir + exp_new + '/' + exp_name + '/net/cv/model_weight.pt',
                    help='path to save the best model')

parser.add_argument('--train_data', type=str,  default=base_dir+"SI_data/New_train_files/train_audio_XRMB_ext_2sec_new.data",  # training with a smaller dev set
                    help='path to training data')

parser.add_argument('--dev_data', type=str,  default=base_dir+"SI_data/New_train_files/dev_audio_XRMB_ext_2sec_new.data",
                    help='path to dev data')

parser.add_argument('--test_data', type=str,  default=base_dir+"SI_data/New_train_files/test_audio_XRMB_ext_2sec_new.data",
                    help='path to testing data')

parser.add_argument('--initialization_data', type=str,  default=base_dir+"SI_data/test_audio_XRMB_ext_2sec.data",
                    help='path to initialization data')

parser.add_argument('--train_random_data', type=str,  default=base_dir+"SI_data/test_audio_XRMB_ext_2sec.data",
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
    print("CUDA ACTIVATED")
else:
    kwargs = {}
    print("CUDA DISABLED")

training_data_path = args.train_data
dev_data_path = args.dev_data
validation_data_path = args.test_data
initialization_data_path = args.initialization_data
train_random_data_path = args.train_random_data

initialization_loader = DataLoader(BoDataset(initialization_data_path),
                           batch_size=args.batch_size,
                           shuffle=False,
                           **kwargs)

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

dev_loader = DataLoader(BoDataset(dev_data_path),
                               batch_size=args.batch_size,
                               shuffle=False,
                               **kwargs)

eval_byPieces_loader = DataLoader(BoDataset(validation_data_path),
                                  batch_size=1,
                                  shuffle=False,
                                  **kwargs)

args.dataset_len = len(train_loader)
args.log_step = args.dataset_len // 2

# E = encoder()
model = synthesis_model()
# print ('. . . . . . . . . . ', list(E.parameters()))

if args.cuda:
    # E.to('cuda')
    model.to('cuda')
    print("converted model to cuda")

# E.load_state_dict(torch.load('tmp/net/Gui_model_weight_E1.pt'))
# D.load_state_dict(torch.load('tmp/net/Gui_model_weight_D1.pt'))


# current_lr = args.lr
# E_optimizer = optim.Adam(E.parameters(), lr=args.encoder_lr)
# E_scheduler = torch.optim.lr_scheduler.ExponentialLR(E_optimizer, gamma=0.5)   # changed from 0.5

model_optimizer = optim.Adam(model.parameters(), lr=args.decoder_lr, eps=1e-4)
model_scheduler = torch.optim.lr_scheduler.ExponentialLR(model_optimizer, gamma=0.5)   # changed from 0.5

# parameters = [p for p in E.parameters()] + [p for p in model.parameters()]
# all_optimizer = optim.Adam(parameters, lr=args.lr)
# all_scheduler = torch.optim.lr_scheduler.ExponentialLR(all_optimizer, gamma=0.5)

# output info

log_print('Experiment start time is {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
log_print('-' * 70)
log_print('Discription:'.format(descripiton))
log_print('-' * 70)
log_print(args)

log_print('-' * 70)
s = 0
# for param in E.parameters():
#     s += np.product(param.size())
for param in model.parameters():
    s += np.product(param.size())

log_print('# of parameters: ' + str(s))

log_print('-' * 70)
log_print('Training Set is {}'.format(training_data_path))
log_print('test set is {}'.format(validation_data_path))
log_print('-' * 70)

### Set the name of the folder
out = "figs/NEW_with_only_1_loss_for_speech_DATE_syn_9TVs_" + str(date.today()) + "_H_" + str(datetime.datetime.now().hour)

if not os.path.exists(out):
    os.makedirs(out)

if not os.path.exists(out + "/loss"):
    os.makedirs(out + "/loss")

print(out)

# generated the spectrograms before anything to make sure there is no issue with already trained weights or something.
spec = getSpectrograms("train")
plotSpectrograms(spec, "before_training_train_data")

'''
       Training
'''
print("Training")
#
trainTogether_newTechnique(epochs=300, init=True, loader_eval="train", save_better_model=True, name='init')

spec = getSpectrograms("train")
plotSpectrograms(spec, "TRAIN_train_data")

spec = getSpectrograms("evaluation")
plotSpectrograms(spec, "TRAIN_evaluation_data")

with open(out_dir + exp_new + '/net/synthesizer_1.pt', 'wb') as f:
    torch.save({'model_state_dict': model.cpu().state_dict(), 'optimizer_state_dict': model_optimizer.state_dict()}, f)
    print("We saved model!")


print("Done.")



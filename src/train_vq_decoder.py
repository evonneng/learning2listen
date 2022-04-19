import argparse
import json
import logging
import numpy as np
import os
import scipy.io as sio

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchvision

from modules.fact_model import setup_model, calc_logit_loss
from vqgan.vqmodules.gan_models import setup_vq_transformer
from utils.base_model_util import *
from utils.load_utils import *


def gather_data(config, X, Y, audio, l_vq_model, patch_size, seq_len, bi):
    """ method to prepare data into proper format for training

    Parameters
    ----------
    X: tensor (B,T1,F)
        Past+current raw speaker motion of sequence length T1
    Y: tensor (B,T2,F)
        Past raw listener motion of sequence length T2
    audio: tensor (B,T3,A)
        Past raw speaker audio of sequence length T3
    l_vq_model:
        pre-trained VQ-VAE model used to discretize the past listener motion and
        decode future listener motion predictions
    patch_size: int
        patch length that we divide seq_len into for the VQ-VAE model
    seq_len: int
        full length of sequence that is taken as input into the VQ-VAE model
    bi: int
        current batch index
    """
    idxStart = bi * config['batch_size']
    speakerData_np = X[idxStart:(idxStart + config['batch_size']), :, :]
    listenerData_np = Y[idxStart:(idxStart + config['batch_size']), :, :]
    audioData_np = audio[idxStart:(idxStart + config['batch_size']), :, :]
    inputs, listener_future, raw_listener, btc = \
        create_data_vq(l_vq_model, speakerData_np, listenerData_np,
                        audioData_np, seq_len,
                        data_type=config['loss_config']['loss_type'],
                        patch_size=patch_size)
    return inputs, listener_future, raw_listener, btc


def generator_train_step(config, epoch, generator, g_optimizer, l_vq_model,
                         train_X, train_Y, train_audio, rng, writer,
                         patch_size, seq_len):
    """ method to prepare data into proper format for training

    see gather_data() for remaining parameter definitions

    Parameters
    ----------
    epoch: int
    generator:
        Predictor model that outputs future listener motion conditioned on past
        listener motion and speaker past+current audio+motion
    g_optimizer:
        optimizer for training the Predictor model
    """

    generator.train()
    batchinds = np.arange(train_X.shape[0] // config['batch_size'])
    totalSteps = len(batchinds)
    rng.shuffle(batchinds)
    avgLoss = 0

    for bii, bi in enumerate(batchinds):
        inputs, listener_future, _, _ = gather_data(config, train_X, train_Y,
                                                    train_audio, l_vq_model,
                                                    patch_size, seq_len, bi)
        prediction = generator(inputs,
                        config['fact_model']['cross_modal_model']['max_mask_len'],
                        -1)
        cut_point = listener_future.shape[1]
        logit_loss = calc_logit_loss(prediction[:,:cut_point,:],
                                     listener_future[:,:cut_point])
        g_loss = logit_loss
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step_and_update_lr()
        avgLoss += g_loss.detach().item()
        if bii % config['log_step'] == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'\
                    .format(epoch, config['num_epochs'], bii, totalSteps,
                            avgLoss / totalSteps, np.exp(avgLoss / totalSteps)))
            avg_Loss = 0
    writer.add_scalar('Loss/train_totalLoss', avgLoss / totalSteps, epoch)


def generator_val_step(config, epoch, generator, g_optimizer, l_vq_model,
                       test_X, test_Y, test_audio, currBestLoss,
                       prev_save_epoch, tag, writer, patch_size, seq_len):
    """ method to validate training of Predictor model

    see generator_train_step() for full parameters definition
    """

    generator.eval()
    batchinds = np.arange(test_X.shape[0] // config['batch_size'])
    totalSteps = len(batchinds)
    testLoss = 0

    for bii, bi in enumerate(batchinds):
        inputs, listener_future, _, _ = gather_data(config, test_X, test_Y,
                                                    test_audio, l_vq_model,
                                                    patch_size, seq_len, bi)
        with torch.no_grad():
            prediction = generator(inputs,
                config['fact_model']['cross_modal_model']['max_mask_len'], -1)
        cut_point = listener_future.shape[1]
        logit_loss = calc_logit_loss(prediction[:,:cut_point,:],
                                     listener_future[:,:cut_point])
        g_loss = logit_loss
        testLoss += g_loss.detach().item()

    testLoss /= totalSteps
    print('val_Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'\
                    .format(epoch, config['num_epochs'], bii, totalSteps,
                            testLoss, np.exp(testLoss)))
    print('----------------------------------')
    writer.add_scalar('Loss/val_totalLoss', testLoss / totalSteps, epoch)

    ## save model if the curent loss is better than previous best
    if testLoss < currBestLoss:
        prev_save_epoch = epoch
        checkpoint = {'config': args.config,
                      'state_dict': generator.state_dict(),
                      'optimizer': {
                        'optimizer': g_optimizer._optimizer.state_dict(),
                        'n_steps': g_optimizer.n_steps,
                      },
                      'epoch': epoch}
        fileName = config['model_path'] + \
                    '{}{}_best.pth'.format(tag, config['pipeline'])
        currBestLoss = testLoss
        torch.save(checkpoint, fileName)
        print('>>>> saving best epoch {}'.format(epoch), testLoss)
    return currBestLoss, prev_save_epoch, testLoss


def main(args):
    """ full pipeline for training the Predictor model """
    rng = np.random.RandomState(23456)
    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)
    print('using config', args.config)
    with open(args.config) as f:
      config = json.load(f)
    tag = config['tag']
    pipeline = config['pipeline']
    writer = SummaryWriter('runs/debug_{}{}'.format(tag, pipeline))
    args.get_attn = False
    currBestLoss = 1e3
    prev_save_epoch = 0
    ## can modify via configs, these are default for released model
    patch_size = 8
    seq_len = 32

    ## setting up the listener VQ-VAE and Predictor models
    # load pre-trained VQ-VAE model
    with open(config['l_vqconfig']) as f:
        l_vqconfig = json.load(f)
    l_model_path = 'vqgan/' + l_vqconfig['model_path'] + \
            '{}{}_best.pth'.format(l_vqconfig['tag'], l_vqconfig['pipeline'])
    l_vq_model, _, _ = setup_vq_transformer(args, l_vqconfig,
                                            load_path=l_model_path)
    for param in l_vq_model.parameters():
        param.requires_grad = False
    l_vq_model.eval()
    vq_configs = {'l_vqconfig': l_vqconfig, 's_vqconfig': None}
    # set up Predictor model
    fileName = config['model_path'] + \
                    '{}{}_best.pth'.format(tag, config['pipeline'])
    load_path = fileName if os.path.exists(fileName) else None
    generator, g_optimizer, start_epoch = setup_model(config, l_vqconfig,
                                                      s_vqconfig=None,
                                                      load_path=load_path)
    generator.train()

    ## training process
    train_X, test_X, train_Y, test_Y, train_audio, test_audio = \
        load_data(config, pipeline, tag, rng, vqconfigs=vq_configs,
                  segment_tag=config['segment_tag'], smooth=True)
    body_mean_dist, body_std_dist = None, None
    for epoch in range(start_epoch, start_epoch + config['num_epochs']):
        print('epoch', epoch, 'num_epochs', config['num_epochs'])
        if epoch == start_epoch+config['num_epochs']-1:
            print('early stopping at:', epoch)
            print('best loss:', currBestLoss)
            break
        generator_train_step(config, epoch, generator, g_optimizer, l_vq_model,
                             train_X, train_Y, train_audio, rng, writer,
                             patch_size, seq_len)
        currBestLoss, prev_save_epoch, g_loss = \
            generator_val_step(config, epoch, generator, g_optimizer, l_vq_model,
                               test_X, test_Y, test_audio, currBestLoss,
                               prev_save_epoch, tag, writer, patch_size, seq_len)
    print('final best loss:', currBestLoss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ar_load', action='store_true')
    args = parser.parse_args()
    main(args)

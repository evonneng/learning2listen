import argparse
import json
import logging
import numpy as np
import os
import scipy.io as sio

import torch
from torch import nn
from torch.autograd import Variable
import torchvision
from torch.utils.tensorboard import SummaryWriter

from vqmodules.gan_models import setup_vq_transformer, calc_vq_loss
import sys
sys.path.append('../')
from utils.load_utils import *


def generator_train_step(config, epoch, generator, g_optimizer, train_X,
                         rng, writer):
    """ Function to do autoencoding training for VQ-VAE

    Parameters
    ----------
    generator:
        VQ-VAE model that takes as input continuous listener and learns to
        outputs discretized listeners
    g_optimizer:
        optimizer that trains the VQ-VAE
    train_X:
        continuous listener motion sequence (acts as the target)
    """

    generator.train()
    batchinds = np.arange(train_X.shape[0] // config['batch_size'])
    totalSteps = len(batchinds)
    rng.shuffle(batchinds)
    avgLoss = avgDLoss = 0
    for bii, bi in enumerate(batchinds):
        idxStart = bi * config['batch_size']
        gtData_np = train_X[idxStart:(idxStart + config['batch_size']), :, :]
        gtData = Variable(torch.from_numpy(gtData_np),
                          requires_grad=False).cuda()
        prediction, quant_loss = generator(gtData, None)
        g_loss = calc_vq_loss(prediction, gtData, quant_loss)
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


def generator_val_step(config, epoch, generator, g_optimizer, test_X,
                       currBestLoss, prev_save_epoch, tag, writer):
    """ Function that validates training of VQ-VAE

    see generator_train_step() for parameter definitions
    """

    generator.eval()
    batchinds = np.arange(test_X.shape[0] // config['batch_size'])
    totalSteps = len(batchinds)
    testLoss = testDLoss = 0
    for bii, bi in enumerate(batchinds):
        idxStart = bi * config['batch_size']
        gtData_np = test_X[idxStart:(idxStart + config['batch_size']), :, :]
        gtData = Variable(torch.from_numpy(gtData_np),
                          requires_grad=False).cuda()
        with torch.no_grad():
            prediction, quant_loss = generator(gtData, None)
        g_loss = calc_vq_loss(prediction, gtData, quant_loss)
        testLoss += g_loss.detach().item()
    testLoss /= totalSteps
    print('val_Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'\
                .format(epoch, config['num_epochs'], bii, totalSteps,
                        testLoss, np.exp(testLoss)))
    print('----------------------------------')
    writer.add_scalar('Loss/val_totalLoss', testLoss / totalSteps, epoch)

    ## save model if curr loss is lower than previous best loss
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
        print('>>>> saving best epoch {}'.format(epoch), testLoss)
        currBestLoss = testLoss
        torch.save(checkpoint, fileName)
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
    currBestLoss = 1e3
    ## can modify via configs, these are default for released model
    seq_len = 32
    prev_save_epoch = 0
    writer = SummaryWriter('runs/debug_{}{}'.format(tag, pipeline))

    ## setting up models
    fileName = config['model_path'] + \
                '{}{}_best.pth'.format(tag, config['pipeline'])
    load_path = fileName if os.path.exists(fileName) else None
    generator, g_optimizer, start_epoch = setup_vq_transformer(args, config,
                                            version=None, load_path=load_path)
    generator.train()

    ## training/validation process
    _, _, train_listener, test_listener, _, _ = \
                    load_data(config, pipeline, tag, rng,
                              segment_tag=config['segment_tag'], smooth=True)
    train_X = np.concatenate((train_listener[:,:seq_len,:],
                              train_listener[:,seq_len:,:]), axis=0)
    test_X = np.concatenate((test_listener[:,:seq_len,:],
                             test_listener[:,seq_len:,:]), axis=0)
    print('loaded listener...', train_X.shape, test_X.shape)
    disc_factor = 0.0
    for epoch in range(start_epoch, start_epoch + config['num_epochs']):
        print('epoch', epoch, 'num_epochs', config['num_epochs'])
        if epoch == start_epoch+config['num_epochs']-1:
            print('early stopping at:', epoch)
            print('best loss:', currBestLoss)
            break
        generator_train_step(config, epoch, generator, g_optimizer, train_X,
                             rng, writer)
        currBestLoss, prev_save_epoch, g_loss = \
            generator_val_step(config, epoch, generator, g_optimizer, test_X,
                               currBestLoss, prev_save_epoch, tag, writer)
    print('final best loss:', currBestLoss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ar_load', action='store_true')
    args = parser.parse_args()
    main(args)

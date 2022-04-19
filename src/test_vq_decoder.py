import argparse
import json
import logging
import numpy as np
import os
import pickle
import scipy.io as sio

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.autograd import Variable

from modules.fact_model import setup_model, calc_logit_loss
from vqgan.vqmodules.gan_models import setup_vq_transformer
from utils.load_utils import *


def run_model(args, config, l_vq_model, generator, test_X, test_Y, test_audio,
              seq_len, patch_size, rng=None):
    """ method to run full model pipeline in autorecursive manner

    Parameters
    ----------
    l_vq_model:
        pre-trained VQ-VAE model used to discretize the past listener motion and
        decode future listener motion predictions
    generator:
        Predictor model that outputs future listener motion conditioned on past
        listener motion and speaker past+current audio+motion
    test_X: tensor (B,T1,F)
        Past+current raw speaker motion of sequence length T1
    test_Y: tensor (B,T2,F)
        Past raw listener motion of sequence length T2
    test_audio: tensor (B,T3,A)
        Past raw speaker audio of sequence length T3
    seq_len: int
        full length of sequence that is taken as input into the VQ-VAE model
    patch_size: int
        patch length that we divide seq_len into for the VQ-VAE model
    rng:
        random number generator for sampling purposes
    """

    batch_size = config['batch_size']
    batchinds = np.arange(test_X.shape[0] // min(test_X.shape[0],batch_size))
    ## set initial masking variables to mask everything
    max_mask_len = config['fact_model']['cross_modal_model']['max_mask_len']
    ## set the point in which we discard the remaining Predictor output
    cut_point = config['fact_model']['listener_past_transformer_config']\
                      ['sequence_length']
    past_cut_point = config['fact_model']['listener_past_transformer_config']\
                           ['sequence_length']*patch_size
    start_t = step_t = patch_size
    output_pred = output_gt = output_probs = None

    for bii, bi in enumerate(batchinds):
        ## define and prepare data into correct format to pass into Predictor
        idxStart = bi * batch_size
        speakerData_np = test_X[idxStart:(idxStart + batch_size), :, :]
        listenerData_np = test_Y[idxStart:(idxStart + batch_size), :, :]
        audioData_np = test_audio[idxStart:(idxStart + batch_size), :, :]
        listenerData_np[:,:seq_len,:] *= 0. ## remove the listener from GT
        prediction, probs, inputs, quant_size = \
            generate_prediction(config, args, l_vq_model, generator,
                                speakerData_np[:,:(seq_len+patch_size),:],
                                listenerData_np[:,:seq_len,:],
                                audioData_np[:,:(seq_len+patch_size)*4,:],
                                seq_len, patch_size, 0, cut_point)
        prediction = torch.cat((inputs['listener_past'],
                                prediction[:,0]), axis=-1)
        probs = torch.cat((torch.zeros((probs.shape[0],
                                        inputs['listener_past'].shape[1],
                                        probs.shape[2])).cuda(),
                                        probs[:,[0],:]), axis=1)

        ## continue for remaining sequence for as long as we have speaker inputs
        for t in range(start_t, test_X.shape[1]-past_cut_point, step_t):
            listener_in = \
                prediction.data[:,int(t/step_t):int((t+seq_len)/step_t)]\
                                                                .cpu().numpy()
            curr_prediction, curr_probs, _ , _= \
                generate_prediction(config, args, l_vq_model, generator,
                                speakerData_np[:,t:(t+seq_len+patch_size),:],
                                listener_in,
                                audioData_np[:,t:(t+(seq_len+patch_size)*4),:],
                                seq_len, patch_size, int(t/step_t), cut_point,
                                btc=quant_size)
            prediction = torch.cat((prediction, curr_prediction[:,0]), axis=1)
            probs = torch.cat((probs, curr_probs[:,[0],:]), axis=1)

        ## once we have the full sequence of output, we decode piece by piece
        decoded_pred = None
        #remove initial gt information
        prediction = prediction[:,quant_size[-1]:]
        for t in range(0, prediction.shape[-1], quant_size[-1]):
            curr_decoded = l_vq_model.module.decode_to_img(
                                prediction[:,t:t+quant_size[-1]], quant_size)
            decoded_pred = curr_decoded if decoded_pred is None \
                            else torch.cat((decoded_pred, curr_decoded), axis=1)
        #re-attach initial gt information (not used in eval)
        prediction = torch.cat((torch.from_numpy(
                                        listenerData_np[:,:seq_len,:]).cuda(),
                                        decoded_pred), dim=1)

        ## calculating upperbound of quantization by decoding and unencoding GT
        decoded_gt = None
        for t in range(0, listenerData_np.shape[1], seq_len):
            tmp = Variable(torch.from_numpy(listenerData_np[:,t:t+seq_len,:]),
                           requires_grad=False).cuda()
            _, gt_logit = l_vq_model.module.get_quant(tmp)
            tmp_decoded = l_vq_model.module.decode_to_img(gt_logit, quant_size)
            decoded_gt = tmp_decoded if decoded_gt is None else \
                            torch.cat((decoded_gt, tmp_decoded), axis=1)

        ## consolidating across all batches
        if output_pred is None:
            output_pred = prediction.data.cpu().numpy()
            output_probs = probs.data.cpu().numpy()
            output_gt = decoded_gt.data.cpu().numpy()
        else:
            output_pred = np.concatenate((output_pred,
                                prediction.data.cpu().numpy()), axis=0)
            output_probs = np.concatenate((output_probs,
                                probs.data.cpu().numpy()), axis=0)
            output_gt = np.concatenate((output_gt,
                                decoded_gt.data.cpu().numpy()), axis=0)

    print('out', output_pred.shape)
    return output_pred, output_probs, output_gt


def generate_prediction(config, args, l_vq_model, generator, test_X,
                        test_Y, test_audio, seq_len, patch_size,
                        mask_point, cut_point, btc=None):
    """ Function to run inputs through Predictor model and to sample outputs

    See above method run_model() for parameter definitions
    """
    ## prepare inputs in proper format to pass through model
    inputs, _, raw_listener, quant_size = \
        create_data_vq(l_vq_model,
                       test_X,
                       test_Y,
                       test_audio,
                       seq_len,
                       data_type=config['loss_config']['loss_type'],
                       patch_size=patch_size, 
                       btc=btc)

    ## run inputs through Predictor model
    with torch.no_grad():
        quant_prediction = generator(inputs,
                config['fact_model']['cross_modal_model']['max_mask_len'],
                mask_point)

    ## sample outputs to obtain probability and predicted logit
    prediction, probs = l_vq_model.module.get_logit(
                                        quant_prediction[:,:cut_point,:],
                                        sample_idx=args.sample_idx)
    return prediction, probs, inputs, quant_size


def save_pred(args, config, tag, pipeline, test_files, unstd_pred, probs=None):
    """ Method to saves predictions and probs to corresponding files """
    ## unstandardize outputs
    B,T,_ = unstd_pred.shape
    preprocess = np.load(os.path.join('vqgan/', config['model_path'],
                                '{}{}_preprocess_core.npz'.format(config['tag'],
                                config['pipeline'])))
    body_mean_Y = preprocess['body_mean_Y']
    body_std_Y = preprocess['body_std_Y']
    test_Y = unstd_pred * body_std_Y + body_mean_Y

    ## save predictions into corresponding files
    for b in range(B):
        for t in range(T):
            vid, _, frame_num = test_files[b,t,:]
            save_base = os.path.join('outputs/', vid,
                                'results/{}predicted/'.format(args.etag+tag))
            if not os.path.exists(save_base):
                os.makedirs(save_base)
            save_path = os.path.join(save_base,
                                '{:08d}.pkl'.format(int(frame_num)))
            data = {'exp': torch.from_numpy(test_Y[b,t,:50]).cuda()[None,...],
                    'pose': torch.from_numpy(test_Y[b,t,50:]).cuda()[None,...]}
            if probs is not None:
                data['prob'] = probs[b,int(t/8),:]
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
    print('done save', test_Y.shape)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.RandomState(23456)
    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)
    seq_len = 32
    patch_size = 8
    num_out = 1024
    with open(args.config) as f:
      config = json.load(f)
    pipeline = config['pipeline']
    tag = config['tag']

    ## setup VQ-VAE model
    with open(config['l_vqconfig']) as f:
        l_vqconfig = json.load(f)
    l_model_path = 'vqgan/' + l_vqconfig['model_path'] + \
        '{}{}_best.pth'.format(l_vqconfig['tag'], l_vqconfig['pipeline'])
    l_vq_model, _, _ = setup_vq_transformer(args, l_vqconfig,
                                            load_path=l_model_path,
                                            test=True)
    l_vq_model.eval()
    vq_configs = {'l_vqconfig': l_vqconfig, 's_vqconfig': None}

    ## setup Predictor model
    load_path = args.checkpoint
    print('> checkpoint', load_path)
    generator, _, _ = setup_model(config, l_vqconfig,
                                  mask_index=0, test=True, s_vqconfig=None,
                                  load_path=load_path)
    generator.eval()

    ## load data
    out_num = 1 if config['data']['speaker'] == 'fallon' else 0
    test_X, test_Y, test_audio, test_files, _ = \
            load_test_data(config, pipeline, tag, out_num=out_num,
                           vqconfigs=vq_configs, smooth=True,
                           speaker=args.speaker, num_out=num_out)

    ## run model and save/eval
    unstd_pred, probs, unstd_ub = run_model(args, config, l_vq_model, generator,
                                            test_X, test_Y, test_audio, seq_len,
                                            patch_size, rng=rng)
    overall_l2 = np.mean(
        np.linalg.norm(test_Y[:,seq_len:,:] - unstd_pred[:,seq_len:,:], axis=-1))
    print('overall l2:', overall_l2)
    if args.save:
        save_pred(args, l_vqconfig, tag, pipeline, test_files, unstd_pred,
                  probs=probs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--speaker', type=str, required=True)
    parser.add_argument('--etag', type=str, default='')
    parser.add_argument('--sample_idx', type=int, default=None)
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()
    print(args)
    main(args)

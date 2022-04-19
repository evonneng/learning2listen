import copy
from modules.base_models import *
from utils.base_model_util import *
from utils.optim import ScheduledOptim

import torch
import torch.nn.functional as F
import json


def calc_logit_loss(pred, target):
    """ Cross entropy loss wrapper """
    loss = F.cross_entropy(pred.reshape(-1, pred.size(-1)), target.reshape(-1))
    return loss


def setup_model(config, l_vqconfig, mask_index=-1, test=False, load_path=None,
                s_vqconfig=None):
    """ Method that sets up Predictor for train/test """

    ## setting model parameters
    quant_factor = l_vqconfig['transformer_config']['quant_factor']
    learning_rate = config['learning_rate']
    print('starting lr', learning_rate)

    ## defining generator model and optimizers
    generator = FACTModel(config['fact_model'],
                          mask_index=mask_index,
                          quant_factor=quant_factor).cuda()
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    generator = nn.DataParallel(generator)
    g_optimizer = ScheduledOptim(
            torch.optim.Adam(generator.parameters(),
                             betas=(0.9, 0.98), eps=1e-09),
                learning_rate,
                config['fact_model']['cross_modal_model']['in_dim'],
                config['warmup_steps'])

    ## if load_path was defined, load model from prev checkpoint to resume
    start_epoch = 0
    if load_path is not None:
        print('loading from checkpoint...', load_path)
        loaded_state = torch.load(load_path,
                                  map_location=lambda storage, loc: storage)
        generator.load_state_dict(loaded_state['state_dict'], strict=True)
        g_optimizer._optimizer.load_state_dict(
                                    loaded_state['optimizer']['optimizer'])
        g_optimizer.set_n_steps(loaded_state['optimizer']['n_steps'])
        start_epoch = loaded_state['epoch']
    else:
        print('starting from scratch...')
    return generator, g_optimizer, start_epoch


## Code adapted from Google [Li 2021]: https://google.github.io/aichoreographer/
class FACTModel(nn.Module):
  """ Predictor model that outputs future listener motion """

  def __init__(self, config, mask_index=-1, quant_factor=None):
    super().__init__()
    self.config = copy.deepcopy(config)

    ## set up listener motion embedding layers
    self.listener_past_transformer = Transformer(
        in_size=self.config['listener_past_transformer_config']\
                           ['hidden_size'],
        hidden_size=self.config['listener_past_transformer_config']\
                               ['hidden_size'],
        num_hidden_layers=self.config['listener_past_transformer_config']\
                                     ['num_hidden_layers'],
        num_attention_heads=self.config['listener_past_transformer_config']\
                                       ['num_attention_heads'],
        intermediate_size=self.config['listener_past_transformer_config']\
                                     ['intermediate_size'])
    self.listener_past_pos_embedding = PositionEmbedding(
        self.config["listener_past_transformer_config"]["sequence_length"],
        self.config['listener_past_transformer_config']['hidden_size'])
    self.listener_past_tok_embedding = nn.Embedding(
        self.config['listener_past_transformer_config']['in_dim'],
        self.config['listener_past_transformer_config']['hidden_size'])

    ## set up speaker audio embedding layers and motion embedding pipeline
    # creating layers the prepare audio and motion modalities independently
    # before cross modal transformer
    dim = self.config['speaker_full_transformer_config']['hidden_size']
    audio_layers = [nn.Sequential(
                        nn.MaxPool1d(4))]
    self.audio_compressor = nn.Sequential(*audio_layers)
    self.audio_projector = nn.Linear(128,dim*2)
    self.audio_full_pos_embedding = PositionEmbedding(
        self.config["speaker_full_transformer_config"]["sequence_length"],
        dim*2)
    self.motion_projector = nn.Linear(56,dim*2)
    self.motion_full_pos_embedding = PositionEmbedding(
        self.config["speaker_full_transformer_config"]["sequence_length"],
        dim*2)
    # creating cross modal transformer that will merge speaker audio and motion
    self.cm_transformer = Transformer(
        in_size=dim*2,
        hidden_size=dim*2,
        num_hidden_layers=2,
        num_attention_heads=self.config['speaker_full_transformer_config']\
                                       ['num_attention_heads'],
        intermediate_size=self.config['speaker_full_transformer_config']\
                                     ['intermediate_size'],
        cross_modal=True)
    # creating post processing layers that will temporally downsample merged
    # speaker embedding
    post_layers = [nn.Sequential(
                   nn.Conv1d(dim*2,dim*2,5,stride=2,padding=2,
                             padding_mode='replicate'),
                   nn.LeakyReLU(0.2, True),
                   nn.BatchNorm1d(dim*2))]
    for _ in range(1, quant_factor):
        post_layers += [nn.Sequential(
                        nn.Conv1d(dim*2,dim*2,5,stride=1,padding=2,
                                  padding_mode='replicate'),
                        nn.LeakyReLU(0.2, True),
                        nn.BatchNorm1d(dim*2),
                        #
                        nn.MaxPool1d(2))]
    self.post_compressor = nn.Sequential(*post_layers)
    self.post_projector = nn.Linear(dim*2, dim)

    ## set up predictor model that takes listener and speaker embeddings and
    # ouptuts future listener motion
    self.cross_modal_layer = CrossModalLayer(self.config['cross_modal_model'])
    self.cross_modal_layer.train()

    self.apply(self._init_weights)
    self.rng = np.random.RandomState(23456)


  def _init_weights(self, module):
    """ Initializes weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

  def gen_mask(self, B, max_mask, mask_index):
      """ Generates a batch of (random) masks based on masking properties

      Parameters
      ----------
      B : int
        the batch dimension (number of masks to generate)
      max_mask: int
        the max index in which we randomly sample. Anything above max_mask is
        never masked
      mask_index: int
        if < 0, we sample a random point from which we mask (during training)
        if > 0, this is the chosen point from which we mask (during testing)
      """
      full_mask = None
      for b in range(B):
        mask = torch.zeros(1,max_mask, max_mask)
        if mask_index < 0:
          num_extra = max_mask * 4  # we want to sample no masking more often
          mask_index = self.rng.randint(0,max_mask+1+num_extra)
        if mask_index > 0:
            mask[:,-mask_index:,:] += 1.
        if full_mask is None:
          full_mask = mask
        else:
          full_mask = torch.cat((full_mask, mask), dim=0)
      return full_mask, mask_index


  def forward(self, inputs, max_mask, mask_index):
    """ Run our full Predictor pipeline

    Given speaker past+current motion+audio, and listener past motion as inputs,
    outputs the corresponding future listener motion.
    """

    ## define masks for train time. no peak on listener, dummy mask for speaker
    dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
    nopeak_mask = {'mask_index': mask_index, 'max_mask': max_mask}
    mask = None
    if max_mask is not None:
        mask, mask_index = self.gen_mask(inputs["listener_past"].shape[0],
                                         max_mask, mask_index)
        mask = mask.unsqueeze(1).cuda()
    nopeak_mask['mask'] = mask
    nopeak_mask['mask_index'] = mask_index

    ## process listener motion segment
    max_context = \
        self.config["listener_past_transformer_config"]["sequence_length"]
    B,T = inputs["listener_past"].shape[0], inputs["listener_past"].shape[1]
    F = self.config["listener_past_transformer_config"]["hidden_size"]
    # process listener differently according to sampled temporal masking condit.
    # < 0: no masking at all. we see everything
    if mask_index < 0:
        listener_past_features = self.listener_past_tok_embedding(
                                    inputs["listener_past"][:,-max_context:])
        listener_past_features = self.listener_past_pos_embedding(
                                    listener_past_features)
        listener_past_features = self.listener_past_transformer(
                                    (listener_past_features, dummy_mask))
    # == 0: complete masking. we see nothing from listener (start condition)
    if mask_index == 0:
        listener_past_features = torch.zeros((B,T,F)).float().cuda()
        listener_past_features = self.listener_past_pos_embedding(
                                    listener_past_features)
    # > 0: random masking. we see only part of listener past (begin condition)
    elif mask_index > 0:
        part_listener_past_features = self.listener_past_tok_embedding(
                                        inputs["listener_past"][:,-mask_index:])
        listener_past_features = torch.zeros((B,T,F)).float().cuda()
        listener_past_features[:,-mask_index:,:] = part_listener_past_features
        listener_past_features = self.listener_past_pos_embedding(
                                        listener_past_features)

    ## process the speaker embedding. no masking. apply cross modal transformer
    audio_full_features = \
        self.audio_compressor(inputs['audio_full'].permute(0,2,1).contiguous())\
                                                  .permute(0,2,1).contiguous()
    audio_full_features = self.audio_projector(audio_full_features)
    audio_full_features = self.audio_full_pos_embedding(audio_full_features)
    motion_full_features = self.motion_projector(inputs['speaker_full'])
    motion_full_features = self.motion_full_pos_embedding(motion_full_features)
    data_features = {'x_a':audio_full_features, 'x_b':motion_full_features}
    speaker_full_features = self.cm_transformer(data_features)
    speaker_full_features = \
        self.post_compressor(speaker_full_features.permute(0,2,1).contiguous())\
                                                  .permute(0,2,1).contiguous()
    speaker_full_features = self.post_projector(speaker_full_features)

    ## process the speaker and listener embeddings together
    output = self.cross_modal_layer(listener_past_features,
                                    speaker_full_features,
                                    nopeak_mask)
    return output

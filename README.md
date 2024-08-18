# Learning to Listen: Modeling Non-Deterministic Dyadic Facial Motion
This repository contains a pytorch implementation of "Learning to Listen: Modeling Non-Deterministic Dyadic Facial Motion"

[![report](https://img.shields.io/badge/arXiv-2204.08451-b31b1b.svg)](https://arxiv.org/abs/2204.08451)
<a href="https://evonneng.github.io/learning2listen/"><img src="https://img.shields.io/badge/project page-github.io-blue"/></a> 

![](src/data/teaser.gif)

This codebase provides:
- train code
- test code
- dataset
- pretrained motion models

The main sections are:
- Overview
- Instalation
- Download Data and Models
- Training from Scratch
- Testing with Pretrained Models

Please note, we will not be providing visualization code for the photorealistic rendering.

# Overview:
We provide models and code to train and test our listener motion models.

See below for sections:
- **Installation**: environment setup and installation for visualization
- **Download data and models**: download annotations and pre-trained models
- **Training from scratch**: scripts to get the training pipeline running from scratch
- **Testing with pretrianed models**: scripts to test pretrained models and save output motion parameters

## Installation:
Tested with cuda/9.0, cudnn/v7.0-cuda.9.0, and python 3.6.11

```
git clone git@github.com:evonneng/learning2listen.git

cd learning2listen/src/
conda create -n venv_l2l python=3.6
conda activate venv_l2l
pip install -r requirements.txt

export L2L_PATH=`pwd`
```

IMPORTANT: After installing torch, please make sure to modify the `site-packages/torch/nn/modules/conv.py` file by commenting out the `self.padding_mode != 'zeros'` line to allow for replicated padding for ConvTranspose1d as shown [here](https://github.com/NVIDIA/tacotron2/issues/182). 

# Download Data and Models:
## Download Data:

Please first download the dataset for the corresponding individuals.

* [Conan Data](http://learning2listen.berkeleyvision.org/conan_data.tar)
* [Devi Data](http://learning2listen.berkeleyvision.org/devi2_data.tar)
* [Fallon Data](http://learning2listen.berkeleyvision.org/fallon_data.tar)
* [Kimmel Data](http://learning2listen.berkeleyvision.org/kimmel_data.tar)
* [Stephen Data](http://learning2listen.berkeleyvision.org/stephen_data.tar)
* [Trevor Data](http://learning2listen.berkeleyvision.org/trevor_data.tar)

Make sure all downloaded .tar files are moved to the directory `$L2L_PATH/data/` (e.g. `$L2L_PATH/data/conan_data.tar`)

Then run the following script.
```
./scripts/unpack_data.sh
```

The downloaded data will unpack into the following directory structure as viewed from `$L2L_PATH`:
```
|-- data/
    |-- conan/
        |-- test/
            |-- p0_list_faces_clean_deca.npy
            |-- p0_speak_audio_clean_deca.npy
            |-- p0_speak_faces_clean_deca.npy
            |-- p0_speak_files_clean_deca.npy
            |-- p1_list_faces_clean_deca.npy
            |-- p1_speak_audio_clean_deca.npy
            |-- p1_speak_faces_clean_deca.npy
            |-- p1_speak_files_clean_deca.npy
        |-- train/
    |-- devi2/
    |-- fallon/
    |-- kimmel/
    |-- stephen/
    |-- trevor/
```

Our dataset consists of 6 different youtube channels named accordingly.
Please see comments in `$L2L_PATH/scripts/download_models.sh` for more details.
For access to the raw videos, please contact Evonne.

## Data Format:
The data format is as described below:

We denote *p0* as the person on the left side of the video, and *p1* as the right side.

* `p0_list_faces_clean_deca.npy` - face features (N x 64 x 184) for when p0 is listener
    * N sequences of length 64. Features of size 184, which includes the deca parameter set of expression (50D), pose (6D), and details (128D). 
* `p0_speak_audio_clean_deca.npy` - audio features (N x 256 x 128) for when p0 is speaking
    * N sequences of length 256. Features of size 128 mel features
* `p0_speak_faces_clean_deca.npy` - face features (N x 64 x 184) for when p0 is speaking
* `p0_speak_files_clean_deca.npy` - file names of the format (N x 64 x 3) for when p0 is speaking 

## Using Your Own Data:
To train and test on your own videos, please follow this process to convert your data into a compatible format:

(Optional) In our paper, we ran preprocessing to figure out when a each person is speaking or listening. We used this information to segment/chunk up our data. We then extracted speaker-only audio by removing listener back-channels.

1. Run [SyncNet](https://github.com/joonson/syncnet_python) on the video to determine who is speaking when. 
2. Then run [Multi Sensory](https://github.com/andrewowens/multisensory) to obtain speaker's audio with all the listener backchannels removed. 

For the main processing, we assuming there are 2 people in the video - one speaker and one listener... 

1. Run [DECA](https://github.com/YadiraF/DECA) to extract the facial expression and pose details of the two faces for each frame in the video. For each person combine the extracted features across the video into a (1 x T x (50+6)) matrix and save to `p0_list_faces_clean_deca.npy` or `p0_speak_faces_clean_deca.npy` files respectively. Note, in concatenating the features, expression comes first.

2. Use `librosa.feature.melspectrogram(...)` to process the speaker's audio into a (1 x 4T x 128) feature. Save to `p0_speak_audio_clean_deca.npy`.


## Download Model:
Please first download the models for the corresponding individual with google drive.

* [Conan Models](https://drive.google.com/file/d/1HlGLMPcshqwdmQvryKPVsYvd9oL2yGML/view?usp=drivesdk)
* [Fallon Models](https://drive.google.com/file/d/1_d4D6T9qflgd15uA3xhtp9wvchWbg9Da/view?usp=drivesdk)
* [Stephen Models](https://drive.google.com/file/d/1gXt2pjpnPItCIfINKCToBacoTI-cVs0W/view?usp=drivesdk)
* [Trevor Models](https://drive.google.com/file/d/1M5y5J3NKhMbzIaU58_Gz8yOFQZuOVhCn/view?usp=drivesdk)

Make sure all downloaded .tar files are moved to the directory `$L2L_PATH/models/` (e.g. `$L2L_PATH/models/conan_models.tar`)

Once downloaded, you can run the follow script to unpack all of the models.

```
cd $L2L_PATH
./scripts/unpack_models.sh
```

We provide person-specific models trained for Conan, Fallon, Stephen, and Trevor.
Each person-specific model consists of 2 models: 1) VQ-VAE pre-trained codebook of motion in `$L2L_PATH/vqgan/models/` and 2) predictor model for listener motion prediction in `$L2L_PATH/models/`. It is important that the models are paired correctly during test time.

In addition to the models, we also provide the corresponding config files that were used to define the models/listener training setup. 

Please see comments in `$L2L_PATH/scripts/unpack_models.sh` for more details.

# Training from Scratch:
Training a model from scratch follows a 2-step process.

1. Train the VQ-VAE codebook of listener motion:
```
# --config: the config file associated with training the codebook
# Includes network setup information and listener information
# See provided config: configs/l2_32_smoothSS.json

cd $L2L_PATH/vqgan/
python train_vq_transformer.py --config <path_to_config_file>
```
Please note, during training of the codebook, it is normal for the loss to increase before decreasing. Typical training was ~2 days on 4 GPUs.

2. After training of the VQ-VAE has converged, we can begin training the predictor model that uses this codebook. 
```
# --config: the config file associated with training the predictor
# Includes network setup information and codebook information
# Note, you will have to update this config to point to the correct codebook.
# See provided config: configs/vq/delta_v6.json

cd $L2L_PATH
python -u train_vq_decoder.py --config <path_to_config_file>
```
Training the predictor model should have a much faster convergance. Typical training was ~half a day on 4 GPUs.

# Testing with Pretrained Models:

```
# --config: the config file associated with training the predictor 
# --checkpoint: the path to the pretrained model
# --speaker: can specify which speaker you want to test on (conan, trevor, stephen, fallon, kimmel)

cd $L2L_PATH
python test_vq_decoder.py --config <path_to_config> --checkpoint <path_to_pretrained_model> --speaker <optional>
```

For our provided models and configs you can run:
``` 
python test_vq_decoder.py --config configs/vq/delta_v6.json --checkpoint models/delta_v6_er2er_best.pth --speaker 'conan'
```

## Visualization
As part of responsible practices, we will not be releasing code for the photorealistic visualization pipeline. 
However, the raw 3D meshes can be rendered using the [DECA renderer](https://github.com/YadiraF/DECA/blob/master/demos/demo_reconstruct.py#L75).

## Potentially Coming Soon
- Visualization of 3D meshes code from saved output


# bibtex
```
@InProceedings{Ng_2022_CVPR,
    author    = {Ng, Evonne and Joo, Hanbyul and Hu, Liwen and Li, Hao and Darrell, Trevor and Kanazawa, Angjoo and Ginosar, Shiry},
    title     = {Learning To Listen: Modeling Non-Deterministic Dyadic Facial Motion},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {20395-20405}
}
```


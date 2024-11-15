from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from utils import AttrDict
from dataset import mel_spectrogram, load_wav
from models import Generator
import soundfile as sf
import librosa
import numpy as np
import time
h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(h):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(h.checkpoint_file_load, device)
    generator.load_state_dict(state_dict_g['generator'])


    filelist = sorted(os.listdir(h.test_input_mels_dir))

    os.makedirs(h.test_output_dir, exist_ok=True)

    generator.eval()
    l=0
    with torch.no_grad():
        starttime = time.time()
        for i, filename in enumerate(filelist):

        
            mel = np.load(os.path.join(h.test_input_mels_dir, filename))
            x = torch.FloatTensor(mel).to(device)
            x=x.unsqueeze(0)
            x=x.transpose(1,2)
          
            logamp_g, pha_g, _, _, y_g = generator(x)
            audio = y_g.squeeze()
          
            audio = audio.cpu().numpy()
            audiolen=len(audio)
            sf.write(os.path.join(h.test_output_dir, filename.split('.')[0]+'.wav'), audio, h.sampling_rate,'PCM_16')

            l+=audiolen


def main():
    print('Initializing Inference Process..')

    config_file = 'config.json'

    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    device = torch.device('cpu')
    inference(h)


if __name__ == '__main__':
    main()


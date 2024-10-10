import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from dataset import Dataset, mel_spectrogram, amp_pha_specturm, get_dataset_filelist
from models import (
    Generator,
    MultiPeriodDiscriminator,
    feature_loss,
    generator_loss,
    discriminator_loss,
    amplitude_loss,
    phase_loss,
    STFT_consistency_loss,
    MultiResolutionDiscriminator,
)
from utils import (
    AttrDict,
    build_env,
    plot_spectrogram,
    scan_checkpoint,
    load_checkpoint,
    save_checkpoint,
)

torch.backends.cudnn.benchmark = True


def train(h):
    
    torch.cuda.manual_seed(h.seed)
    device = torch.device("cuda:{:d}".format(0))
    generator = Generator(h).to(device)
    
    os.makedirs(h.checkpoint_path, exist_ok=True)
    
    steps = 0
    generator.load_state_dict(torch.load("checkpoints/g_final_1")["generator"])
    
    
    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    
    
    training_filelist = get_dataset_filelist(
        h.input_training_wav_list
    )

    
    trainset = Dataset(
        training_filelist,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        h.meloss,
        n_cache_reuse=0,
        shuffle=True,
        device=device,
    )

    train_loader = DataLoader(
        trainset,
        num_workers=h.num_workers,
        shuffle=True,
        sampler=None,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True,
    )

    generator.train()
    

    for epoch in range(0, h.training_epochs):
        
        start = time.time()
       

        for i, batch in enumerate(train_loader):
            
            start_b = time.time()
            
            x, logamp, pha, rea, imag, y, meloss, inv_mel, pghid = map(
                lambda x: x.to(device, non_blocking=True), batch
            )
            
            y = y.unsqueeze(1)
            
            logamp_g, pha_g, rea_g, imag_g, y_g = generator(x)
            
            y_g_mel = mel_spectrogram(
                y_g.squeeze(1),
                h.n_fft,
                h.num_mels,
                h.sampling_rate,
                h.hop_size,
                h.win_size,
                h.fmin,
                h.meloss,
            )

            optim_g.zero_grad()
            L_A = amplitude_loss(logamp, logamp_g)

            L_IP, L_GD, L_PTD = phase_loss(pha, pha_g, h.n_fft, pha.size()[-1])

            L_P = L_IP + L_GD + L_PTD

            _, _, rea_g_final, imag_g_final = amp_pha_specturm(
                y_g.squeeze(1), h.n_fft, h.hop_size, h.win_size
            )
            
            L_C = STFT_consistency_loss(rea_g, rea_g_final, imag_g, imag_g_final)
            L_R = F.l1_loss(rea, rea_g)
            L_I = F.l1_loss(imag, imag_g)
            L_S = L_C + 2.25 * (L_R + L_I)
        
            L_Mel = F.l1_loss(meloss, y_g_mel)
          
            L_G = 45 * L_A + 100 * L_P + 20 * L_S + 45 * L_Mel

            L_G.backward()
            optim_g.step()

            if steps % 1 == 0:
                with torch.no_grad():
                    A_error = amplitude_loss(logamp, logamp_g).item()
                    IP_error, GD_error, PTD_error = phase_loss(
                        pha, pha_g, h.n_fft, pha.size()[-1]
                    )
                    IP_error = IP_error.item()
                    GD_error = GD_error.item()
                    PTD_error = PTD_error.item()
                    C_error = STFT_consistency_loss(
                        rea_g, rea_g_final, imag_g, imag_g_final
                    ).item()
                    R_error = F.l1_loss(rea, rea_g).item()
                    I_error = F.l1_loss(imag, imag_g).item()
                    Mel_error = F.l1_loss(x, y_g_mel).item()

                print(
                    "Steps : {:d}, Gen Loss Total : {:4.3f}, Amplitude Loss : {:4.3f}, Instantaneous Phase Loss : {:4.3f}, Group Delay Loss : {:4.3f}, Phase Time Difference Loss : {:4.3f}, STFT Consistency Loss : {:4.3f}, Real Part Loss : {:4.3f}, Imaginary Part Loss : {:4.3f}, Mel Spectrogram Loss : {:4.3f}, s/b : {:4.3f}".format(
                        steps,
                        L_G,
                        A_error,
                        IP_error,
                        GD_error,
                        PTD_error,
                        C_error,
                        R_error,
                        I_error,
                        Mel_error,
                        time.time() - start_b,
                    )
                )


            checkpoint_path = "{}/g_final_2".format(h.checkpoint_path, steps)
            save_checkpoint(checkpoint_path, {"generator": generator.state_dict()})
            
        
            steps += 1

        print("Time taken for epoch {} is {} sec\n".format(epoch + 1, int(time.time() - start)))


def main():

    config_file = "config.json"

    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(config_file, "config.json", h.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
    else:
        pass

    train(h)


if __name__ == "__main__":
    main()

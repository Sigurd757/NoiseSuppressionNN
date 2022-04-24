import os
import torch
import librosa
import soundfile as sf
import json
import numpy as np
import random
import time
from torch.utils.data import DataLoader
from torch import nn
import argparse
from tensorboardX import SummaryWriter
from numpy.linalg import norm
from data_preparation.data_preparation import FileDateset
from model.Baseline import Base_model
from model.ops import pytorch_LSD
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train_data', default="/media/qcm/HardDisk1/jsonfile/denoise/train220316.json")
    parser.add_argument('--val_data', default="/media/qcm/HardDisk1/jsonfile/denoise/eval220316.json")
    parser.add_argument('--test_data', default="/media/qcm/HardDisk1/jsonfile/denoise/eval220316.json")
    parser.add_argument('--checkpoints_dir', default="./checkpoints/AEC_baseline")
    parser.add_argument('--event_dir', default="./event_file/AEC_baseline")
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--output_dir', default="./test_output/")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print("GPU是否可用：", torch.cuda.is_available())  # True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ###########    保存输出的地址    ############
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    ################################
    #          实例化模型          #
    ################################
    model = Base_model().to(device)  # 实例化模型

    ###############################
    # 创建优化器 Create optimizers #
    ###############################
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, )
    # lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.min_lr, last_epoch=-1)
    
    # ###########    加载模型检查点   ############
    start_epoch = 0
    if args.model_name:
        print("加载模型：",  args.model_name)
        checkpoint = torch.load(args.model_name)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint['epoch']
        lr_schedule.load_state_dict(checkpoint['lr_schedule'])
        
    model.eval() #test mode
    
    clean_speech_list = []
    noise_signal_list = []
    with open(args.test_data) as fid:
        for line in fid:
            load_dict = json.loads(line)
            clean_speech_list.append(load_dict["audio_filepath"])
            noise_signal_list.append(load_dict["noise_file"])
        
    def add_noise(clean, noise, snr):
        if len(noise) > len(clean):
            noise = noise[:len(clean)]
        else:
            times = len(clean) // len(noise)
            noise = np.tile(noise, times)
            padding = [0] * (len(clean) - len(noise))
            noise = np.hstack([noise, padding])

        noise = noise / norm(noise) * norm(clean) / (10.0 ** (0.1 * snr))
        noise = noise.astype(np.float32)
        mix = clean + noise
        return clean, noise, mix
    
    def spectrogram(wav):
        S1 = librosa.stft(wav, n_fft=512, hop_length=160, win_length=320, window="hann", center=False)
        S = torch.from_numpy(S1)
        magnitude = torch.abs(S)        # 振幅
        phase = torch.exp(1j * torch.angle(S))  # 相位
        return magnitude, phase
    
    sample_num = len(clean_speech_list)
    
    with torch.no_grad():
        for test_idx, clean_speech_wav_file in enumerate(clean_speech_list):
            clean_speech_wav, _ = librosa.load(path=clean_speech_list[test_idx], sr=16000)
            noise_wav, _ = librosa.load(path=noise_signal_list[test_idx], sr=16000)
            #add noise
            snr= random.uniform(-5, 10)
            clean_speech_wav_proc, noise_wav_proc, mix_speech_wav_proc = add_noise(clean_speech_wav, noise_wav, snr)

            # 带噪语音 振幅，相位 （F, T）,F为频点数，T为帧数
            mix_speech_magnitude, mix_speech_phase = spectrogram(mix_speech_wav_proc)  # torch.Size([161, 999])
            # 噪声 振幅，相位
            noise_wav_magnitude, noise_wav_phase = spectrogram(noise_wav_proc)
            # 纯净语音 振幅，相位
            clean_speech_magnitude, clean_speech_phase = spectrogram(clean_speech_wav_proc)
            
            mix_speech_magnitude = mix_speech_magnitude.to(device)
            print(mix_speech_magnitude.shape)
            mix_speech_magnitude = torch.unsqueeze(mix_speech_magnitude, 0)
            print(mix_speech_magnitude.shape)
            mix_speech_phase =mix_speech_phase.to(device)
            mix_speech_phase = torch.unsqueeze(mix_speech_phase, 0)
            
            # 前向传播
            val_ped_mask = model(mix_speech_magnitude)
            val_denoise_spectrum = val_ped_mask * mix_speech_magnitude * mix_speech_phase
            val_denoise_spectrum = val_denoise_spectrum.cpu()
            val_denoise_spectrum = val_denoise_spectrum.detach().numpy().squeeze()
            
            print(val_denoise_spectrum.shape)
            denoise_wav_name = clean_speech_wav_file.split('/')[-1].replace(".wav", "_denoised.wav")
            val_pred_denoise_wav = librosa.istft(val_denoise_spectrum, hop_length=160, win_length=320)
            sf.write(os.path.join(args.output_dir, denoise_wav_name), val_pred_denoise_wav, 16000)
            
            mix_wav_name = clean_speech_wav_file.split('/')[-1].replace(".wav", "_mixd.wav")
            sf.write(os.path.join(args.output_dir, mix_wav_name), mix_speech_wav_proc, 16000)



if __name__ == "__main__":
    main()

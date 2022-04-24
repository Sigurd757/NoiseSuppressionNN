import glob
import os
import torch.nn.functional as F
import torch
import librosa
import json
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy.linalg import norm
from tqdm import tqdm


class FileDateset(Dataset):
    def __init__(self, dataset_path="/media/qcm/HardDisk1/jsonfile/denoise/train220316.json", noiseset_path ="/media/qcm/HardDisk1/code/AEC_DeepModel/noise.txt", fs=16000, win_length=320, fft_len=512, win_inc=160, mode="train"):
        self.fs = fs
        self.win_length = win_length
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.mode = mode
        self.stride = 4096
        self.dimension = 8192

        clean_speech_list = []
        noise_signal_list = []

        with open(dataset_path) as fid:
            for line in fid:
                load_dict = json.loads(line)
                clean_speech_list.append(load_dict["audio_filepath"])
                #noise_signal_list.append(load_dict["noise_file"])
        
        with open(noiseset_path) as fid_noise:
            for line_noise in fid_noise:
                line_noise = line_noise.strip()
                if os.path.isdir(line_noise):
                    # print(line_noise)
                    for (root, dirs, files) in os.walk(line_noise):
                        for f_noise in files:
                            noise_file = os.path.join(root, f_noise)
                            if noise_file.endswith(".wav"):
                                noise_signal_list.append(noise_file)

        self.clean_speech_list = clean_speech_list
        self.noise_signal_list = noise_signal_list
        self.clean_list_len = len(self.clean_speech_list)
        self.noise_list_len = len(self.noise_signal_list)
        # self.clean_wave_data = self.split(self.clean_speech_list)
        # self.noise_wave_data = self.split(self.noise_signal_list)
        # self.clean_wave_len = len(self.clean_wave_data)
        # self.noise_wave_len = len(self.noise_wave_data)
        # print("speech num:", self.clean_wave_len)
        # print("noise num:", self.noise_wave_len)

    def spectrogram(self, wav):
        """
        :param wav_path: 音频路径
        :return: 返回该音频的振幅和相位
        """
        # wav, _ = torchaudio.load(wav_path)
        # wav = wav.squeeze()

        # if len(wav) < 160000:
        #     wav = F.pad(wav, (0,160000-len(wav)), mode="constant",value=0)

        # S = torch.stft(wav, n_fft=self.win_length, hop_length=self.win_length//2,
        #                win_length=self.win_length, window=torch.hann_window(window_length=self.win_length),
        #                center=False, return_complex=True)   # (*, F,T)
        S1 = librosa.stft(wav, n_fft=self.fft_len, hop_length=self.win_inc, win_length=self.win_length, window="hann", center=False)
        S = torch.from_numpy(S1)

        magnitude = torch.abs(S)        # 振幅
        phase = torch.exp(1j * torch.angle(S))  # 相位
        return magnitude, phase

    def split(self, wav_path_list):
        wave_data = []
        for wav_path in tqdm(wav_path_list):
            wav, _ = librosa.load(path=wav_path, sr=self.fs)
            wav_length = len(wav)  # 音频长度
            if wav_length < self.stride:  # 如果语音长度小于4096
                continue
            if wav_length < self.dimension:  # 如果语音长度小于8192
                diffe = self.dimension - wav_length
                wav_frame = np.pad(wav, (0, diffe), mode="constant")
                wave_data.append(wav_frame)
            else:  # 如果音频大于 8192
                start_index = 0
                while True:
                    if start_index + self.dimension > wav_length:
                        break
                    wav_frame = wav[start_index:start_index + self.dimension]
                    wave_data.append(wav_frame)
                    start_index += self.stride
        return wave_data

    def subsample(self,  wav_path, sub_sample_length=64000, start_position: int = -1, return_start_position=False):
        """
        Randomly select fixed-length data from 

        Args:
            data: **one-dimensional data**
            sub_sample_length: how long
            start_position: If start index smaller than 0, randomly generate one index

        """
        data, _ = librosa.load(path=wav_path, sr=self.fs)
        assert np.ndim(data) == 1, f"Only support 1D data. The dim is {np.ndim(data)}"
        length = len(data)

        if length > sub_sample_length:
            if start_position < 0:
                start_position = np.random.randint(length - sub_sample_length)
            end = start_position + sub_sample_length
            data = data[start_position:end]
        elif length < sub_sample_length:
            data = np.append(data, np.zeros(sub_sample_length - length, dtype=np.float32))
        else:
            pass

        assert len(data) == sub_sample_length

        if return_start_position:
            return data, start_position
        else:
            return data
    
    
    def __getitem__(self, item):
        """__getitem__是类的专有方法，使类可以像list一样按照索引来获取元素
        :param item: 索引
        :return:  按 索引取出来的 元素
        """
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

        # clean_speech_wav = self.clean_wave_data[item]
        # noise_wav = self.noise_wave_data[random.randint(0, self.noise_wave_len - 1)]
        clean_speech_wav = self.subsample(self.clean_speech_list[item])
        noise_wav = self.subsample(self.noise_signal_list[random.randint(0, self.noise_list_len - 1)])
        # noise_wav = self.subsample(self.noise_signal_list[item])
        #add noise
        snr= random.uniform(-5, 10)
        clean_speech_wav_proc, noise_wav_proc, mix_speech_wav_proc = add_noise(clean_speech_wav, noise_wav, snr)
        
        clean_speech_wav_proc = torch.from_numpy(clean_speech_wav_proc)
        noise_wav_proc = torch.from_numpy(noise_wav_proc)
        mix_speech_wav_proc = torch.from_numpy(mix_speech_wav_proc)
        clean_speech_wav_proc = torch.unsqueeze(clean_speech_wav_proc, 0)
        noise_wav_proc = torch.unsqueeze(noise_wav_proc, 0)
        mix_speech_wav_proc = torch.unsqueeze(mix_speech_wav_proc, 0)
        '''
        # 带噪语音 振幅，相位 （F, T）,F为频点数，T为帧数
        mix_speech_magnitude, mix_speech_phase = self.spectrogram(mix_speech_wav_proc)  # torch.Size([161, 999])
        # 噪声 振幅，相位
        noise_wav_magnitude, noise_wav_phase = self.spectrogram(noise_wav_proc)
        # 纯净语音 振幅，相位
        clean_speech_magnitude, clean_speech_phase = self.spectrogram(clean_speech_wav_proc)

        _eps = torch.finfo(torch.float).eps  # 防止分母出现0
        mask_IRM = torch.sqrt(clean_speech_magnitude ** 2/(mix_speech_magnitude ** 2+_eps))  # IRM，模型输出
        '''

        #return mix_speech_magnitude, mask_IRM, clean_speech_magnitude
        return mix_speech_wav_proc, noise_wav_proc, clean_speech_wav_proc


    def __len__(self):
        """__len__是类的专有方法，获取整个数据的长度"""
        return len(self.clean_speech_list)


if __name__ == "__main__":
    train_set = FileDateset(dataset_path="/media/qcm/HardDisk1/jsonfile/denoise/train220316.json")
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True)

    for x, y, z in train_loader:
        print(x.shape)  # torch.Size([64, 322, 999])
        print(y.shape)  # torch.Size([64, 161, 999])
        print(x.type())
        print(y.type())
        print(z.type())


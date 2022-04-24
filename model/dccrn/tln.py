import torch
import torch.nn as nn
import math
import librosa
import sys
import os
import torch.nn.functional as F
from loss import si_snr, sd_snr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# print(os.path.dirname(os.path.abspath(__file__)))
from show import show_params, show_model

mel_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 , 21, 22, 23, 24, 25, 26, 27, 28, 
                            29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 
                            56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 
                            85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 136, 140, 144, 148,
                            152, 156, 160, 164, 169, 174, 179, 184, 189, 194, 199, 204, 209, 214, 220, 226, 232, 238, 244, 250, 257]     #128

def feq2mel(inputs):
    n_mel = len(mel_index) - 1
    outputs = []
    for i in range(n_mel):
        average = inputs[:,mel_index[i]:mel_index[i+1], :].mean(1,True)
        outputs.append(average)
    return torch.cat(outputs, dim=1)

def mel2feq(inputs):
    outputs = []
    for i in range(len(mel_index) - 1):
        for j in range(mel_index[i+1] - mel_index[i]):
            outputs.append(inpus[:, i:i+1, :])
    return torch.cat(outputs, dim=1)

def wav_norm(x):
    b, _, t = x.size()
    s = 0
    f = None
    o = []
    while s<t:
        m = torch.mean(torch.abs(x[:, :, s:s+160]), dim=2, keepdim=True)
        if f is not None:
            m = 0.02 * m + 0.98 * f
        f = m
        x[:, :, s:s + 160] = x[:, :, s:s + 160] / (f * 100.)
        o.append(f * 100.)
        s+=160
        
    return x, o

def wav_inorm(x, f):
    b, _, t = x.size()
    s = 0
    i = 0
    while s<t:
        x[:, :, s:s + 160] = x[:, :, s:s + 160] * f[i]
        s += 160
        i += 1
    return x

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        
    def forward(self, x):
        return x * torch.sigmoid(x)

class DFSMNRes(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        k_size,
        pad,
    ):
        super(DFSMNRes, self).__init__()
        
        self.d1 = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU()
        )
        
        self.in_liner = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        )
        
        self.hidden_conv = nn.Sequential(
            nn.ConstantPad1d([pad, k_size - pad -1], 0),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k_size, padding=0, groups=hidden_dim)
        )
        
        self.norm_act = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU()
        )
        
        self.weight = nn.Parameter(torch.Tensor([0]), requires_grad=True)
        
        self.outliner = nn.Sequential(
            nn.Conv1d(hidden_dim, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(output_dim)
        )
        
        self.act = nn.PReLU()
        
    def forward(self, inputs, hidden=None):
        
        out = self.d1(inputs)
        
        out = self.in_liner(out)
        
        hid = self.hidden_conv(out)
        
        out_p = self.norm_act(out + hid)
        
        if hidden is not None:
            out_p = hidden * self.weight + out_p
        
        out = self.outliner(out_p)
        
        out = self.act(out + inputs)
        
        return out, out_p
    
class DfsmnLargeDecoder(nn.Module):
    def __init__(self, n, d, h, k ,p):
        super(DfsmnLargeDecoder, self).__init__()
        
        self.dfsmn_layers = nn.ModuleList()
        
        self.dfsmn_layers_n = n
        
        for i in range(self.dfsmn_layers_n):
            self.dfsmn_layers.append(DFSMNRes(d, h, d, k, p))
            
            
    def forward(self, inputs):
        hidden = None
        for i in range(self.dfsmn_layers_n):
            inputs, hidden = self.dfsmn_layers[i](inputs, hidden=hidden)
        return inputs
    
class TLN_10ms(nn.Module):
    def __init__(self, ) :
        super(TLN_10ms, self).__init__()
        
        self.conv1 = torch.nn.Sequential(
            nn.ConstantPad1d([0, 320], 0),
            nn.Conv1d(1, 320, 320, stride=160, bias=False,)
        )
        
        self.norm = nn.BatchNorm1d(320)
        
        self.dfsmn1 = DfsmnLargeDecoder(5, 320, 160, 40, 39)
        
        self.conv2 = torch.nn.Sequential(
            nn.Conv1d(320, 320, 1, bias=True,),
            nn.Sigmoid()
        )
        
        self.conv3 = nn.Conv1d(320, 320, 1, stride=1, bias=False,)
        
        show_params(self)
        
    def forward(self, x):
        x1 = self.conv1(x)
        print(x1.shape)
        x2 = self.norm(x1)
        print(x2.shape)
        x2 = self.dfsmn1(x2)
        print(x2.shape)
        x2 = self.conv2(x2)
        print(x2.shape)
        x2 = x2*x1
        x2 = self.conv3(x2)
        x2 = x2.transpose(1, 2)
        x2_1 = x2[:, :, :160]
        x2_2 = torch.cat((x2[:,:1,:160], x2[:, :-1, 160:]), dim=1)
        #print(x2.shape)
        #print(x2_1.shape)
        #print(x2_2.shape)
        x2 = torch.reshape(x2_1 + x2_2, [x.size(0), 1, -1])
        #print(x2.shape)
        return x2
    
    def loss(self, inputs, labels, loss_mode='SI-SNR'):
       
        if loss_mode == 'MSE':
            b,t,d = inputs.shape 
            #gth_spec = self.stft(labels)
            return F.mse_loss(inputs, labels, reduction='mean')*d

        elif loss_mode == 'SI-SNR':
            #return -torch.mean(si_snr(inputs, labels))
            return -(si_snr(inputs, labels))
        elif loss_mode == 'SD-SNR':
            #return -torch.mean(si_snr(inputs, labels))
            return -(sd_snr(inputs, labels))
        elif loss_mode == 'MAE':
            gth_spec, gth_phase = self.stft(labels) 
            b,d,t = inputs.shape 
            return torch.mean(torch.abs(inputs-gth_spec))*d

    
if __name__ == '__main__':
    torch.manual_seed(10)
    torch.autograd.set_detect_anomaly(True)
    inputs = torch.randn([10,1,16000*4]).clamp_(-1,1)
    labels = torch.randn([10,1,16000*4]).clamp_(-1,1)
    print(inputs.shape)
    
    net = TLN_10ms()
    outputs = net(inputs)
    print(outputs.shape)
    outputs = outputs[:,:,160:]
    print(outputs.shape)
    loss = net.loss(outputs, labels, loss_mode='SI-SNR')
    print(loss)

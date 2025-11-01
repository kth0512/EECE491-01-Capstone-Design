import torch
import torch.nn as nn

def awgn_channel(x, snr_db):
    signal_power = torch.mean(x.pow(2))
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise_std = torch.sqrt(noise_power)
    noise = torch.randn_like(x) * noise_std
    return x + noise
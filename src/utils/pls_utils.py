import torch
import torch.nn as nn
import numpy as np 
import os
import random
import matplotlib.pyplot as plt

def cn01(shape):
    a = np.random.randn(*shape).astype(np.float32)
    b = np.random.randn(*shape).astype(np.float32)
    return (a + 1j * b).astype(np.complex64) / np.sqrt(2.0).astype(np.float32)

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    return v if n < eps else v / n

def sample_channels(Nt, Ne):
    hb = cn01((Nt, 1))      # Bob 채널
    He = cn01((Nt, Ne))     # Eve 채널
    return hb, He

def precoder_mrt(hb_hat):
    return normalize(hb_hat.astype(np.complex64)).astype(np.complex64)

def projector_null(hb_hat):
    w = normalize(hb_hat.astype(np.complex64))
    return np.eye(len(hb_hat), dtype=np.complex64) - (w @ w.conj().T)

def pls_an_channel(s, alpha, hb, He,
                   Ptot=1.0, SNRdB=10.0,
                   imperfect_csi=False, csi_err_var=0.0):
    """
    단일 스트림 AN 기반 PLS 채널 (Bob과 Eve의 수신 심볼 반환)
    s : (Nsym,) complex64
    """
    Nt, Ne = He.shape[0], He.shape[1]
    sigma2 = np.float32(Ptot * 10**(-SNRdB / 10))

    if imperfect_csi:
        e = cn01((Nt, 1)) * np.sqrt(np.float32(csi_err_var))
        hb_hat = (hb + e).astype(np.complex64)
    else:
        hb_hat = hb.astype(np.complex64)

    w = precoder_mrt(hb_hat)      # (Nt,1)
    Pnull = projector_null(hb_hat) # (Nt,Nt)

    Ps  = np.float32((1.0 - alpha) * Ptot)
    Pan = np.float32(alpha * Ptot)
    Nsym = s.shape[0]

    z = cn01((Nt, Nsym))
    n_mat = Pnull @ z
    n_mat = n_mat / (np.linalg.norm(n_mat, axis=0, keepdims=True) + 1e-12)

    x = (np.sqrt(Ps).astype(np.float32) * (w @ s.reshape(1, -1))).T \
        + np.sqrt(Pan).astype(np.float32) * n_mat.T

    # --- Bob 수신 ---
    nb = (np.sqrt(sigma2 / 2.0) * cn01((Nsym, 1))).flatten()
    yb = (hb.conj().T @ x.T).flatten() + nb  # (Nsym,)

    # --- Eve 수신 ---
    ne = (np.sqrt(sigma2 / 2.0) * cn01((Nsym, Ne)))
    Ye = (He.conj().T @ x.T).T + ne      # (Nsym, Ne)
    y_e = Ye.sum(axis=1)                 # (Eve가 여러 안테나의 신호를 합침)

    return yb.astype(np.complex64), y_e.astype(np.complex64)

def latent_to_complex_np(latent_torch: torch.Tensor):
    """ [B, D] real 텐서를 [B, D/2] complex numpy 배열로 변환 """
    z = latent_torch.detach().cpu().numpy().astype(np.float32)
    B, D = z.shape
    assert D % 2 == 0, "latent_dim must be even to map to complex symbols"
    real = z[:, 0::2]
    imag = z[:, 1::2]
    s = (real + 1j * imag).astype(np.complex64)
    return s

def complex_to_latent_torch(s_np: np.ndarray, device):
    """ [B, Nsym] complex numpy 배열을 [B, 2*Nsym] real 텐서로 변환 """
    real = np.real(s_np).astype(np.float32)
    imag = np.imag(s_np).astype(np.float32)
    z = np.stack([real, imag], axis=-1).reshape(real.shape[0], -1)
    z_torch = torch.from_numpy(z).to(device)
    return z_torch

def an_pls_channel(latent_vector: torch.Tensor,
                   snr_db: float,
                   alpha: float = 0.5,
                   Nt: int = 4,
                   Ne: int = 1,
                   Ptot: float = 1.0,
                   imperfect_csi: bool = False,
                   csi_err_var: float = 0.0):
    """
    AWGN 채널을 대체하는 메인 PLS 채널 함수.
    (Bob과 Eve의 latent 텐서를 모두 반환)
    """
    device = latent_vector.device
    B, D = latent_vector.shape

    s_batch = latent_to_complex_np(latent_vector)
    B, Nsym = s_batch.shape

    yb_list = []
    ye_list = []

    for b in range(B):
        hb, He = sample_channels(Nt, Ne)

        yb, y_e = pls_an_channel(
            s_batch[b],
            alpha=alpha,
            hb=hb,
            He=He,
            Ptot=Ptot,
            SNRdB=snr_db,
            imperfect_csi=imperfect_csi,
            csi_err_var=csi_err_var
        )
        yb_list.append(yb)
        ye_list.append(y_e)

    yb_arr = np.stack(yb_list, axis=0) # [B, Nsym]
    ye_arr = np.stack(ye_list, axis=0) # [B, Nsym]

    latent_bob = complex_to_latent_torch(yb_arr, device=device)
    latent_eve = complex_to_latent_torch(ye_arr, device=device)

    return latent_bob, latent_eve
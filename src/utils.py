import skimage
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch_msssim import SSIM
from sklearn.feature_selection import mutual_info_regression
from processing import compress_iq
import torch.nn as nn
import os
from model import CIDNet3D, CxUnet_RB



def get_psnr(y_pred, y_true, g_compr=False, return_per_sample=False):
    """Gamma compression of the pred and true, by default calculates on the log compressed image"""
    if g_compr:
        y_pred = compress_iq(y_pred, mode='log')
        y_true = compress_iq(y_true, mode='log')
    else: 
        y_pred = y_pred.abs() 
        y_true = y_true.abs()

    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu()

    y_pred_np = y_pred.cpu().numpy()
    y_true_np = y_true.cpu().numpy()

    psnr_values = []
    for i in range(y_pred.shape[0]):
        pred_i = y_pred_np[i,0] # get the volume [192,192,192]
        true_i = y_true_np[i,0]
        psnr = skimage.metrics.peak_signal_noise_ratio(true_i, pred_i, data_range=np.max(true_i))
        psnr_values.append(psnr)
    if return_per_sample:
        return psnr_values
    else:
        return np.mean(psnr_values)

def get_ssim(y_pred, y_true, g_compr=False):
    """Calculate the ssim for 3D images per batch, by first doing lo compression 
    input size must be tensors [B, 1, D, H, W]
    works for validation only (because data was detahced)
    https://github.com/VainF/pytorch-msssim"""
    if g_compr:
        # Perform log compression  (Log compression returns torch.Tensor, float32, CPU or original device)
        y_pred = compress_iq(y_pred, mode='log')
        y_true = compress_iq(y_true, mode='log')
    else:
        y_pred = y_pred.abs()
        y_true = y_true.abs()

    # Get ms ssim
    B, C, D, H, W = y_pred.shape
    ssim_scores = []
    ssim_module = SSIM(data_range=1.0, size_average=True, channel=1, spatial_dims=3)

    for b in range(B):
        ssim_val = ssim_module(y_pred[b:b+1], y_true[b:b+1])  # shape: [1, 1, D, H, W]
        ssim_scores.append(ssim_val.item())

    return sum(ssim_scores) / B

# Mutual Information
def mutual_info(y_pred, y_true):
    y_pred = y_pred.abs().cpu().numpy().flatten()
    y_true = y_true.abs().cpu().numpy().flatten()
    return mutual_info_regression(y_true.reshape(-1, 1), y_pred)[0]


class ComplexMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(ComplexMSELoss, self).__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # Compute squared difference on real and imaginary parts
        loss = (y_true.real - y_pred.real) ** 2 + (y_true.imag - y_pred.imag) ** 2

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:  # 'none'
            return loss
        
class ComplexL1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(ComplexL1Loss, self).__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # Compute absolute difference on real and imaginary parts
        loss = torch.abs(y_true.real - y_pred.real) + torch.abs(y_true.imag - y_pred.imag)

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:  # 'none'
            return loss
        
class ComplexTVLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(ComplexTVLoss, self).__init__()
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction

    def forward(self, x):
        # TV on real part
        real = x.real
        imag = x.imag

        loss_real = (
            torch.abs(real[:, :, :-1, :, :] - real[:, :, 1:, :, :]).sum() +
            torch.abs(real[:, :, :, :-1, :] - real[:, :, :, 1:, :]).sum() +
            torch.abs(real[:, :, :, :, :-1] - real[:, :, :, :, 1:]).sum()
        )

        # TV on imaginary part
        loss_imag = (
            torch.abs(imag[:, :, :-1, :, :] - imag[:, :, 1:, :, :]).sum() +
            torch.abs(imag[:, :, :, :-1, :] - imag[:, :, :, 1:, :]).sum() +
            torch.abs(imag[:, :, :, :, :-1] - imag[:, :, :, :, 1:]).sum()
        )

        loss = loss_real + loss_imag

        if self.reduction == 'mean':
            return loss / x.numel()
        elif self.reduction == 'sum':
            return loss
        else:  # 'none'
            return loss 

class ComplexL1TVLoss(nn.Module):
    def __init__(self, tv_weight=0.01, reduction='mean'):
        super().__init__()
        self.l1 = ComplexL1Loss(reduction=reduction)
        self.tv = ComplexTVLoss(reduction=reduction)
        self.tv_weight = tv_weight

    def forward(self, y_pred, y_true):
        l1_loss = self.l1(y_pred, y_true)
        tv_loss = self.tv(y_pred)
        return l1_loss + self.tv_weight * tv_loss

class ComplexMSETVLoss(nn.Module):
    def __init__(self, tv_weight=0.01, reduction='mean'):
        super().__init__()
        self.mse = ComplexMSELoss(reduction=reduction)
        self.tv = ComplexTVLoss(reduction=reduction)
        self.tv_weight = tv_weight

    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)
        tv_loss = self.tv(y_pred)
        return mse_loss + self.tv_weight * tv_loss
    
class ComplexMSE_SSIMLoss(nn.Module):
    def __init__(self, alpha=1, beta=0.5,  reduction='mean'):
        super(ComplexMSE_SSIMLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = ComplexMSELoss(reduction=reduction)
        self.ssim_module = SSIM(data_range=1.0, size_average=True, channel=1, spatial_dims=3)

    def forward(self, y_pred, y_true):
        # MSE on complex-valued data
        mse = self.mse_loss(y_pred, y_true)

        # SSIM on envelope
        y_pred_mag = y_pred.abs()
        y_true_mag = y_true.abs()
        ssim_loss = 1.0 - (self.ssim_module(y_pred_mag, y_true_mag))

        # Total combined loss
        total_loss = self.alpha * mse + self.beta * ssim_loss
        return total_loss
    
def load_best_model_if_exists(model, path, device):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded best model from {path}")
    else:
        print(f"Warning: Best model file not found at {path}")

def get_model_by_name(name):
    models = {
        # "CNN_nn": CNN_nn,
        # "Unet3Dcx": Unet3Dcx,
        "CIDNet3D": CIDNet3D,
        "CxUnet_RB": CxUnet_RB,
        # "CIDNet3D_L": CIDNet3D_L, 
        # "CNN_AMU": CNN_AMU,
    }
    model_class = models.get(name)
    if model_class is None:
        raise ValueError(f"Unknown model name: {name}. Available models: {list(models.keys())}")
    return model_class

# Modified get_loss_by_name
def get_loss_by_name(name, **kwargs): # Accept arbitrary keyword arguments
    losses = {
        "ComplexMSELoss": ComplexMSELoss,
        "ComplexL1Loss": ComplexL1Loss,
        "ComplexMSETVLoss": ComplexMSETVLoss,
        "ComplexL1TVLoss": ComplexL1TVLoss,
        "ComplexMSE_SSIMLoss": ComplexMSE_SSIMLoss,
    }
    loss_class = losses.get(name)
    if loss_class is None:
        raise ValueError(f"Unknown loss function name: {name}. Available losses: {list(losses.keys())}")
    return loss_class(**kwargs) # Pass all kwargs directly to the constructor

def get_scheduler_by_name(name, optimizer, **kwargs): # Accept arbitrary kwargs
    schedulers = {
        "StepLR": torch.optim.lr_scheduler.StepLR,
        "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
        "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
    }
    scheduler_class = schedulers.get(name)
    if scheduler_class is None:
        raise ValueError(f"Unknown scheduler name: {name}. Available schedulers: {list(schedulers.keys())}")
    return scheduler_class(optimizer, **kwargs) # Pass kwargs to constructor

def resolve_config_references(cfg, root_cfg=None):
    """Resolve references like ${training.epochs} (try Hydra later)"""
    if root_cfg is None:
        root_cfg = cfg
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if isinstance(v, str) and v.startswith('${') and v.endswith('}'):
                path = v[2:-1]
                parts = path.split('.')
                resolved_value = root_cfg
                for part in parts:
                    resolved_value = resolved_value[part]
                cfg[k] = resolved_value
            else:
                cfg[k] = resolve_config_references(v, root_cfg)
    elif isinstance(cfg, list):
        for i, item in enumerate(cfg):
            cfg[i] = resolve_config_references(item, root_cfg)
    return cfg
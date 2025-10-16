import torch
import numpy as np
import torchio as tio
import torch

random_affine = tio.RandomAffine(
    scales=(0.95, 1.05),         # Random scaling in each direction (±5%)
    degrees=10,                  # Random rotation up to ±10° on each axis
    translation=5,               # Random translation up to ±5 voxels
    center='image',              # Rotate/scale about the center of the image
    p=0.5                        # 50% probability of applying
)


def compress_iq(iq_data, mode='log', dynamic_range_dB=50, gamma=0.5):
    """works for 3D tensors and numpy arrays and batches as well"""
    is_tensor = isinstance(iq_data, torch.Tensor)
    orig_device = None
    if is_tensor:
        orig_device = iq_data.device
        iq_data = iq_data.cpu().numpy()

    envelope = np.abs(iq_data)
    envelope /= (np.amax(envelope, axis=(-3, -2, -1), keepdims=True) + 1e-12)  # Normalize

    if mode == 'log':
        epsilon = 1e-10
        log_image = 20 * np.log10(envelope + epsilon)
        log_image += dynamic_range_dB
        log_image[log_image < 0] = 0
        result = log_image.astype(np.float32)

    elif mode == 'gamma':
        gamma_image = envelope ** gamma
        gamma_image = np.clip(gamma_image * 255, 0, 255).astype(np.float32)
        result = gamma_image

    else:
        raise ValueError(f"Unsupported compression mode: {mode}. Choose 'log' or 'gamma'.")

    if is_tensor:
        result = torch.tensor(result, dtype=torch.float32, device=orig_device)

    return result


# https://github.com/tristan-deep/dehazing-diffusion/blob/main/processing.py
def mu_law_compress(iq_data, mu=255):
    """Apply μ-law companding to complex I/Q data
    Takes the magnitud and compresses the range of the signal 
    μ detemrines the amount of compression applied"""
    magnitude = torch.abs(iq_data)  #  magnitude
    phase = torch.angle(iq_data)  #  phase

    # Apply μ-law to magnitude (no need of sign because the abs is always positive)
    compressed_mag = torch.log(1 + mu * magnitude) / torch.log(torch.tensor(1 + mu))

    # Reconstruct compressed complex number
    compressed_iq = compressed_mag * torch.exp(1j * phase)
    return compressed_iq

def mu_law_expand(compressed_iq, mu=255):
    """Expand μ-law to compressed I/Q data"""
    compressed_mag = torch.abs(compressed_iq)  # Get magnitude
    phase = torch.angle(compressed_iq)  # Get phase

    # Apply inverse μ-law to magnitude
    expanded_mag = ((1 + mu) ** compressed_mag - 1) / mu

    # Reconstruct expanded complex signal
    expanded_iq = expanded_mag * torch.exp(1j * phase)
    return expanded_iq

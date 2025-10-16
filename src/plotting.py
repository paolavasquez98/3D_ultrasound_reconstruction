import matplotlib.pyplot as plt
import numpy as np
from processing import compress_iq
import torch

def plot_input_pred_tar(input_img, prediction, target):
    """Plot input, prediction, and target volumes side by side. Funcion used in testing and training"""
    depth_idx = input_img.shape[0] // 2  # Middle depth slice

    # Extract the same depth slice across all three volumes
    input_slice = input_img[:, :, depth_idx]
    pred_slice = prediction[:, :, depth_idx]
    target_slice = target[:, :, depth_idx]

    # Create the subplot grid
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot Input
    axes[0].imshow(input_slice, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title("Input")

    # Plot Prediction
    axes[1].imshow(pred_slice, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title("Prediction")

    # Plot Target
    axes[2].imshow(target_slice, cmap='gray')
    axes[2].axis('off')
    axes[2].set_title("Target")

    plt.tight_layout()
    return fig

def visualize_tensor(tensor, title="Feature Map", slice_idx=0):
    """
    Visualize the magnitude of a complex 3D tensor slice [B, C, D, H, W].
    """
    mag = compress_iq(tensor)  # get magnitude of complex tensor
    mag = mag[0, 0]  # take first batch and first channel
    slice_2d = mag[slice_idx].detach().cpu().numpy()

    fig, axes = plt.subplots(figsize=(6, 6))
    axes.imshow(slice_2d, cmap='gray')
    axes.set_title(title)
    axes.axis('off')
    plt.tight_layout()

    return fig

def plot_input_pred_3axis(input_img, prediction):
    """Plot input and prediction images in 3 axes (Axial, Coronal, Sagittal).
    tHIS FUNCTION IS USED FOR INFERENCE WHERE NO gt IS AVAILABLE"""
    if input_img.ndim != 3 or prediction.ndim != 3:
        raise ValueError("Input and prediction images must be 3D volumes for 3-axis plotting.")

    # Get dimensions
    z_dim, x_dim, y_dim = input_img.shape

    # Physical dimensions in mm
    # here is to consider the beamforming values
    z_extent = 60 #130
    x_extent = 30 #50
    y_extent = 30 #50

    dz = z_extent / z_dim
    dx = x_extent / x_dim
    dy = y_extent / y_dim

    z = np.linspace(0, z_extent, z_dim)
    x = np.linspace(0, x_extent, x_dim)
    y = np.linspace(0, y_extent, y_dim)

    # Slice indices
    z_idx = z_dim // 2
    x_idx = x_dim // 2
    y_idx = y_dim // 2
    depth_target = 72 # mm i my case
    # z_idx = int(round(depth_target / dz))

    # Extract slices
    input_axial = input_img[z_idx, :, :] #z_idx
    input_coronal = input_img[:, x_idx, :]
    input_sagittal = input_img[:, :, y_idx]

    pred_axial = prediction[z_idx, :, :] # z_idx
    pred_coronal = prediction[:, x_idx, :]
    pred_sagittal = prediction[:, :, y_idx]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Input - Axial (X × Y)
    axes[0, 0].imshow(input_axial, cmap='gray', extent=[y[0], y[-1], x[-1], x[0]])
    axes[0, 0].set_title("Input (Axial View)")
    axes[0, 0].set_xlabel("y (mm)")
    axes[0, 0].set_ylabel("x (mm)")

    # Input - Coronal (X × Z)
    axes[0, 1].imshow(input_coronal, cmap='gray', extent=[y[0], y[-1], z[-1], z[0]])
    axes[0, 1].set_title("Input (Coronal View)")
    axes[0, 1].set_xlabel("y (mm)")
    axes[0, 1].set_ylabel("z (mm)")

    # Input - Sagittal (Y × Z)
    axes[0, 2].imshow(input_sagittal, cmap='gray', extent=[x[0], x[-1], z[-1], z[0]])
    axes[0, 2].set_title("Input (Sagittal View)")
    axes[0, 2].set_xlabel("x (mm)")
    axes[0, 2].set_ylabel("z (mm)")

    # Prediction - Axial
    axes[1, 0].imshow(pred_axial, cmap='gray', extent=[y[0], y[-1], x[-1], x[0]])
    axes[1, 0].set_title("Prediction (Axial View)")
    axes[1, 0].set_xlabel("y (mm)")
    axes[1, 0].set_ylabel("x (mm)")

    # Prediction - Coronal
    axes[1, 1].imshow(pred_coronal, cmap='gray', extent=[y[0], y[-1], z[-1], z[0]])
    axes[1, 1].set_title("Prediction (Coronal View)")
    axes[1, 1].set_xlabel("y (mm)")
    axes[1, 1].set_ylabel("z (mm)")

    # Prediction - Sagittal
    axes[1, 2].imshow(pred_sagittal, cmap='gray', extent=[x[0], x[-1], z[-1], z[0]])
    axes[1, 2].set_title("Prediction (Sagittal View)")
    axes[1, 2].set_xlabel("x (mm)")
    axes[1, 2].set_ylabel("z (mm)")

    # Optional: turn off axes if you prefer cleaner look
    for ax in axes.ravel():
        ax.axis('on')

    plt.tight_layout()
    return fig

def plot_input_pred(input_img, prediction):
    # Determine the depth index for slicing
    # Assuming input_img is (depth, width, height) after squeezing
    if input_img.ndim == 3: # If it's a 3D volume
        depth_idx = input_img.shape[0] // 2  # Middle depth slice
        input_slice = input_img[depth_idx, :, :] # Slice first along depth
        pred_slice = prediction[depth_idx, :, :]
    elif input_img.ndim == 2: # If it's already a 2D slice
        input_slice = input_img
        pred_slice = prediction
    else:
        raise ValueError(f"Unsupported image dimensions for plotting: {input_img.ndim}")


    # Create the subplot grid (now 1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Input
    axes[0].imshow(input_slice, cmap='gray')
    axes[0].axis('off')
    axes[0].set_title("Input")

    # Plot Prediction
    axes[1].imshow(pred_slice, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title("Prediction")

    plt.tight_layout()
    return fig

def plot_slice(volume, slice_idx=None):
    """Plots a slice of volume """
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().numpy() 
    assert volume.ndim == 3, "Input volume must be 3D (Depth, Height, Width)"
    depth, height, width = volume.shape  # [192, 192, 192]
    if slice_idx is None:
        slice_idx = width // 2  # Select the middle slice by default
    side_slice = volume[:, slice_idx, :]  # Extract the sagittal slice
    # Plot the side view
    plt.figure(figsize=(6, 6))
    plt.imshow(side_slice, cmap='gray', aspect='auto')
    plt.axis('off')
    plt.title(f"Volume Slice {slice_idx}")
    plt.show()
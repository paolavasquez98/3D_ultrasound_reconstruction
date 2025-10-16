import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import get_psnr, get_ssim
from processing import compress_iq, mu_law_expand
from training_pb import sliding_window_inference
from plotting import plot_input_pred_tar, plot_input_pred_3axis, plot_input_pred

def test_model(model, dataloader, criterion, device, norm_method):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    psnr_values, ssim_values = [], []
    image_logs = []
    input_vols, output_vols, target_vols = [], [], []

    with torch.no_grad():
        tqdm_bar = tqdm(dataloader, desc="Testing", leave=False)
        
        for batch_idx, (dwi_data, conv_data) in enumerate(tqdm_bar):
            dwi_data, conv_data = dwi_data.to(device), conv_data.to(device)

            outputs = model(dwi_data) # output Shape: [batch_size, 1, 192, 192, 192]

            # Save some data for visualization
            if len(image_logs) < 6:
                pred_img, gt_img, input_img = outputs, conv_data, dwi_data

                if norm_method in ['compand', 'z_compand']:
                    expanded_input = mu_law_expand(input_img).cpu().numpy()
                    expanded_pred = mu_law_expand(pred_img).cpu().numpy()
                    expanded_tar = mu_law_expand(gt_img).cpu().numpy()
                    
                    inp = np.mean(expanded_input[0], axis=0).squeeze()                    
                    pred = expanded_pred[0, 0].squeeze()                    
                    tar = expanded_tar[0, 0].squeeze()

                elif norm_method in ['max', 'mean_std', 'std']:
                    input_np = input_img.cpu().numpy()
                    pred_np = pred_img.cpu().numpy()
                    gt_np = gt_img.cpu().numpy()

                    inp = np.mean(input_np[0], axis=0).squeeze()
                    pred = pred_np[0, 0].squeeze()
                    tar = gt_np[0, 0].squeeze()

                else:
                    raise ValueError(f"Unknown normalization method: {norm_method}")
                
                # Prepare visualizations
                img = plot_input_pred_tar(np.abs(inp), np.abs(pred), np.abs(tar))
                log_img = plot_input_pred_tar(
                    compress_iq(inp, mode='log', dynamic_range_dB=60),
                    compress_iq(pred, mode='log', dynamic_range_dB=60),
                    compress_iq(tar, mode='log', dynamic_range_dB=60)
                )

                # momentaneo para guardar
                input_vols.append(compress_iq(inp, mode='log', dynamic_range_dB=60))
                output_vols.append(compress_iq(pred, mode='log', dynamic_range_dB=60))
                target_vols.append(compress_iq(tar, mode='log', dynamic_range_dB=60))

                # Save slices for visualization
                image_logs.append({
                "image": img,
                "log_image": log_img,
                # "gamma_image": gamma_img
                })

            # Compute loss
            loss = criterion(outputs, conv_data)
            running_loss += loss.item()

            # Compute PSNR and SSIM
            psnr_value = get_psnr(outputs, conv_data)
            ssim_value = get_ssim(outputs, conv_data)
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)

            tqdm_bar.set_postfix(
                loss=running_loss / (batch_idx + 1),
                psnr=np.mean(psnr_values),
                ssim=np.mean(ssim_values)
        )
            wandb.log({
                "test_loss_mse": loss,
                "test_psnr": psnr_value,
                "test_ssim": ssim_value
            })

    for i, img_dict in enumerate(image_logs):
        wandb.log({
            f"Sample_{i}": [
                wandb.Image(img_dict["image"], caption="Input/Prediction/GT (raw)"),
                wandb.Image(img_dict["log_image"], caption="Log compressed"),
            ]
        })

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_loss = running_loss / len(dataloader)
    print(f"\nTest Loss: {avg_loss:.4f}")
    print(f"\nTest PSNR: {avg_psnr:.2f} dB")
    print(f"\nTest SSIM: {avg_ssim:.4f}")

    return avg_loss, avg_psnr, avg_ssim, input_vols, output_vols, target_vols

def predict_model(model, dataloader, device, norm_method):
    model.eval()  # Set model to evaluation mode
    
    image_logs = []
    input_vols, output_vols, raw_vols = [], [], []

    with torch.no_grad():
        tqdm_bar = tqdm(dataloader, desc="Predicting", leave=False)
        
        for batch_idx, dwi_data in enumerate(tqdm_bar): 
            dwi_data = dwi_data.to(device)

            outputs = model(dwi_data)

            if len(image_logs) < 6: 
                # Initialize inp and pred to None
                inp = None
                pred = None

                # Ensure these are converted to numpy and squeezed *within* each block
                # as the normalization method dictates.
                if norm_method == 'compand':
                    expanded_input = mu_law_expand(dwi_data[0]).squeeze().cpu().numpy()
                    inp = np.mean(expanded_input, axis=0) # Mean across the 9 diverging wave inputs
                    expanded_pred = mu_law_expand(outputs[0, 0]).squeeze().cpu().numpy()
                    pred = expanded_pred
                elif norm_method == 'max':
                    inp = np.mean(dwi_data[0].cpu().numpy(), axis=0).squeeze()
                    pred = outputs[0, 0].cpu().numpy().squeeze()
                elif norm_method == 'mean_std':
                    inp = np.mean(dwi_data[0].cpu().numpy(), axis=0).squeeze()
                    pred = outputs[0, 0].cpu().numpy().squeeze()
                elif norm_method == 'z_compand':
                    expanded_input = mu_law_expand(dwi_data[0]).squeeze().cpu().numpy()
                    inp = np.mean(expanded_input, axis=0)
                    expanded_pred = mu_law_expand(outputs[0, 0]).squeeze().cpu().numpy()
                    pred = expanded_pred
                elif norm_method == 'std':
                    inp = np.mean(dwi_data[0].cpu().numpy(), axis=0).squeeze()
                    pred = outputs[0, 0].cpu().numpy().squeeze()
                else:
                    # This case should ideally raise an error as it's an unhandled normalization
                    raise ValueError(f"Unknown normalization method: {norm_method}")
                
                if inp is not None and pred is not None:
                    # Use plot_input_pred as target is None
                    img = plot_input_pred_3axis(np.abs(inp), np.abs(pred))
                    log_img = plot_input_pred_3axis(
                        compress_iq(inp, mode='log', dynamic_range_dB=60),
                        compress_iq(pred, mode='log', dynamic_range_dB=60)
                    )
                    
                    input_vols.append(compress_iq(inp, mode='log', dynamic_range_dB=60))
                    output_vols.append(compress_iq(pred, mode='log', dynamic_range_dB=60))
                    raw_vols.append(inp)

                    image_logs.append({
                        "image": img,
                        "log_image": log_img,
                    })

    for i, img_dict in enumerate(image_logs):
        wandb.log({
            f"Sample_{i}": [
                wandb.Image(img_dict["image"], caption="Input/Prediction/GT (raw)"),
                wandb.Image(img_dict["log_image"], caption="Log compressed"),
            ]
        })

    return input_vols, output_vols
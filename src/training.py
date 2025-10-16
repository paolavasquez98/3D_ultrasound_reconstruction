import torch
import os
import wandb
import numpy as np
from tqdm import tqdm
from utils import get_psnr, get_ssim
from processing import compress_iq
from plotting import plot_input_pred_tar

# TRAIN ONE EPOCH
def train(model, train_loader, optimizer, criterion, device, save_predictions=False, num_samples=5,  accumulation_steps=2):
    model.train()  # Set model to training mode
    epoch_loss = 0.0
    saved_data = 0
    image_logs = []

    optimizer.zero_grad()
    for batch_idx, (dwi_data, conv_data) in enumerate(tqdm(train_loader, desc="Training")):
        dwi_data, conv_data = dwi_data.to(device), conv_data.to(device)

        train_out = model(dwi_data)
        train_loss = criterion(train_out, conv_data) / accumulation_steps
        
        train_loss.backward()
        # optimizer.step()

        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            # Update weights
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += train_loss.item() * accumulation_steps # i get the true total loss per batch

        # save some predictios
        if save_predictions and saved_data < num_samples:
            pred_img = train_out[0].detach().cpu().numpy()
            gt_img = conv_data[0].detach().cpu().numpy()
            input_img = dwi_data[0].detach().cpu().numpy()

            pred = pred_img.squeeze(0)
            tar = gt_img.squeeze(0)
            inp = np.mean(input_img, axis=0, keepdims=True).squeeze(0)

            log_img = plot_input_pred_tar(
                    compress_iq(inp, mode='log', dynamic_range_dB=60),
                    compress_iq(pred, mode='log', dynamic_range_dB=60),
                    compress_iq(tar, mode='log', dynamic_range_dB=60)
                )


            image_logs.append({"log_image": log_img})
            saved_data += 1

    avg_loss = epoch_loss / len(train_loader)
    return avg_loss, image_logs

# VALIDATION
def validate(model, val_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    epoch_loss = 0.0
    psnr_values = []
    ssim_values = []
    with torch.no_grad(): 
        for batch_idx, (dwi_data, conv_data) in enumerate(tqdm(val_loader, desc="Validating")):
            dwi_data, conv_data = dwi_data.to(device), conv_data.to(device)

            val_out = model(dwi_data) # Shape: [batch_size, 1, 192, 192, 192]

            val_loss = criterion(val_out, conv_data)
            epoch_loss += val_loss.item()

            # Compute PSNR and SSIM per batch
            psnr_value = get_psnr(val_out, conv_data)
            ms_ssim_value = get_ssim(val_out, conv_data)

            psnr_values.append(psnr_value)
            ssim_values.append(ms_ssim_value)

    # average for the epoch
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_loss = epoch_loss / len(val_loader)

    return avg_loss, avg_psnr, avg_ssim


# FULL TRAINING FUNCTION
def train_and_validate(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs, device, patience):
    best_val_loss = float("inf")  # Initialize with a high value
    best_model_path = os.path.join(wandb.run.dir, 'best_model.pth')
    early_stopping_counter = 0
    best_image_logs = []
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        train_loss, image_logs = train(model, train_loader, optimizer, criterion, device, save_predictions=True, num_samples=5, accumulation_steps=2)
        val_loss, avg_psnr, avg_ssim = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Log everything to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_psnr": avg_psnr,
            "val_ssim": avg_ssim,
        })

        print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, PSNR = {avg_psnr:.2f} dB, SSIM = {avg_ssim:.3f}")
        if val_loss < best_val_loss:
            # Save images 
            best_image_logs = image_logs[:5]  # Save for later
            image_logs = []  # Clear after saving, to avoid old data mixing
            for i, img_dict in enumerate(best_image_logs):
                wandb.log({
                    f"BestSample_E{epoch}_#{i}": [
                        wandb.Image(img_dict["log_image"], caption="Log compressed")
                        ]
                    })

            print(f"New best model found at epoch {epoch} with Val Loss: {val_loss:.6f}")
            best_val_loss = val_loss
            best_psnr = avg_psnr
            best_ssim = avg_ssim
            # Save the best model
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved to {best_model_path}")
            wandb.save(best_model_path)  # optional: upload model to wandb
            wandb.config.update({"best_model_path": best_model_path})
            artifact = wandb.Artifact(f"{wandb.run.name}-best-model", type="model")
            artifact.add_file(best_model_path)
            wandb.log_artifact(artifact)

            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_psnr"] = best_psnr
            wandb.run.summary["best_ssim"] = best_ssim
            print(f"\nBest Val Loss: {best_val_loss:.6f} | PSNR: {best_psnr:.2f} dB | SSIM: {best_ssim:.3f}")
            early_stopping_counter = 0  # Reset counter
        else:
            early_stopping_counter += 1
            print(f"No improvement in validation loss. Early stopping counter: {early_stopping_counter}/{patience}")
            if early_stopping_counter >= patience:
                wandb.run.summary["stopped_epoch"] = epoch
                print(f"Stopping early after {epoch} epochs due to no improvement.")
                break

    print("\nTraining complete!")
    return best_image_logs, best_model_path
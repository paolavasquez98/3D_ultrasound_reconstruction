import sys
import os
import torch
import yaml
import wandb
import numpy as np
import random
import argparse
import traceback 
from torch.utils.data import DataLoader, Subset
from datetime import datetime
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "src"))

from dataset import BeamformDataset
from utils import get_loss_by_name, get_model_by_name, resolve_config_references, get_scheduler_by_name, load_best_model_if_exists
from training import train_and_validate
from testing import test_model

# --- Argparse for config file ---
def train_function():
    try:
        parser = argparse.ArgumentParser(description="Train a complex model using a config file.")
        parser.add_argument('--config_path', type=str, required=True,
                            help='Path to the YAML configuration file.')
        args = parser.parse_args()

        # --- Load configuration ---
        base_config = {}
        if args.config_path:
            with open(args.config_path, 'r') as f:
                base_config = yaml.safe_load(f)

        config = resolve_config_references(base_config)

        # Generate a timestamped name
        timestamp = datetime.now().strftime("%m%d%H%M")
        auto_name = f"{config['model']['name']}_{timestamp}"

        wandb.init(
            project=config['wandb']['project'], # Or get from base_config: base_config.get('wandb', {}).get('project')
            entity=config['wandb']['entity'],    # Or get from base_config
            dir=config['wandb']['dir'], # Or get from base_config
            name= auto_name,
            job_type=config['wandb']['job_type'],
            group=config['wandb'].get('group'),
            config=base_config, # This sets initial values/defaults
            save_code=True,
        )

        config = wandb.config
        mutable_config = dict(wandb.config)
        # Resolve references in config
        resolved_config = resolve_config_references(mutable_config)
        # Update wandb.config with the resolved values
        wandb.config.update(resolved_config, allow_val_change=True)
        config = wandb.config

        # Set seeds for reproducibility
        seed = config.get('seed', 42)       
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # --------------------- Dataset and Dataloader Setup ---------------------
        dataset = BeamformDataset(
            config['dataset']['train_path'],
            normalization=config['dataset']['normalization'],
            patch_size=config['dataset']['patch_size']
        )

        idx = list(range(len(dataset)))
        train_idx, test_idx = train_test_split(
            idx, 
            test_size=config['dataset'].get('test_split_ratio', 0.1),
            random_state=config['dataset'].get('split_random_state', 42)
        )
        train_data = Subset(dataset, train_idx)
        val_data = Subset(dataset, test_idx)

        train_dataloader = DataLoader(
            train_data,
            batch_size=config['dataloader']['batch_size'],
            shuffle=config['dataloader']['train_shuffle'],
            pin_memory=config['dataloader']['pin_memory'],
            drop_last=config['dataloader']['drop_last']
        )
        val_dataloader = DataLoader(
            val_data,
            batch_size=config['dataloader']['batch_size'],
            shuffle=config['dataloader']['val_shuffle'],
            pin_memory=config['dataloader']['pin_memory']
        )

        # --------------------- Model Setup ---------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device {device}")

        # Use the helper functions
        model = get_model_by_name(config['model']['name'])(**config['model'].get('args', {})).to(device)
        print(model)
        optimizer_class = getattr(torch.optim, config['training']['optimizer']) # Use getattr for optimizer class
        optimizer = optimizer_class(model.parameters(), **config['training'].get('optimizer_args', {}))
        
        scheduler = get_scheduler_by_name(config['training']['scheduler'], optimizer,
                                        **config['training'].get('scheduler_args', {}))
        criterion = get_loss_by_name(config['training']['criterion'],
                                    **config['training'].get('criterion_args', {}))

        model_params = sum(p.numel() for p in model.parameters())

        wandb.config.update({
            "optimizer_class_name": optimizer.__class__.__name__,
            "loss_class_name": criterion.__class__.__name__,
            "architecture_class_name": model.__class__.__name__,
            "scheduler_class_name": scheduler.__class__.__name__,
            "model_parameters": model_params,
            "train_data_size": len(train_data),
            "val_data_size": len(val_data),
            "data_basename": os.path.basename(config['dataset']['train_path']),
        }, allow_val_change=True)

        wandb.watch(model, log="all", log_freq=100)

        print("All configurations loaded and logged to wandb")

        # Train and Validate
        training_results, best_model_path = train_and_validate(
            model, train_dataloader, val_dataloader, optimizer, criterion, scheduler,
            config['training']['epochs'], device, config['training']['patience']
        )

        print(f"Experiment configuration saved to wandb: {wandb.run.dir}")

        # ---------------------- Testing ---------------------
        test_dataset = BeamformDataset(
            config['dataset']['test_path'],
            normalization=config['dataset']['normalization'],
            mode=config['dataset']['mode'],
            patch_size=config['dataset']['patch_size']
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config['dataloader']['test_batch_size'],
            shuffle=config['dataloader'].get('test_shuffle', False)
        )

        load_best_model_if_exists(model, best_model_path, device)

        loss, psnr, ssim, input_vols, output_vols, target_vols = test_model(
            model, test_dataloader, criterion, device, config['dataset']['normalization']
        )
        wandb.finish() 

    except Exception as e:
        print(f"An unexpected error occurred: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

        if wandb.run:
            error_details = f"An unhandled error occurred: {type(e).__name__}: {e}\n\nTraceback:\n{traceback.format_exc()}"
            wandb.run.alert(
                title="Experiment Failed",
                text=error_details,
                level=wandb.AlertLevel.ERROR
            )
            wandb.finish(exit_code=1)
        sys.exit(1)

if __name__ == "__main__":

    train_function()
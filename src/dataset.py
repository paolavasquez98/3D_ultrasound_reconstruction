import h5py
import numpy as np
from torch.utils.data import Dataset
# import torchvision.transforms as transforms
import torch
import random
from processing import mu_law_compress
import torchio as tio
        
class BeamformDataset(Dataset):
    def __init__(self, path, normalization='mean_std', transform=None, indices=None, mode='Train', patch_size=None):
        self.path = path
        self.normalization = normalization.strip().lower()
        self.transform = transform
        self.db = None

        self.mode = mode.strip().lower() 
        if self.mode not in ['train', 'test', 'inference']:
            raise ValueError(f"Invalid mode: {mode}. Expected 'train', 'test', or 'inference'.")

        # Patch-based loading logic
        self.patch_size = patch_size # Can be None for full volume, or a tuple (D, H, W)
        if self.patch_size is not None and not isinstance(self.patch_size, (tuple, list)):
            raise ValueError("patch_size must be a tuple/list (D, H, W) or None.")

        with h5py.File(self.path, 'r') as db: # open file once
            full_length = db["dwi"].shape[0]
            self.has_target = 'tar' in db

        self.indices = indices if indices is not None else list(range(full_length))

    def __len__(self):
            return len(self.indices)

    def __getitem__(self, index):
        if self.db is None:
            self.db = h5py.File(self.path, 'r', swmr=True)

        real_index = self.indices[index]

        try:
            dwi_data = self.db["dwi"][real_index]
            dwi_data = torch.tensor(dwi_data, dtype=torch.cfloat)
            conv_data = None # Initialize to None

            if self.mode == 'train':
                if not self.has_target:
                    raise RuntimeError(f"Mode is 'train' but 'tar' dataset not found in {self.path}. Cannot train without target.")
                conv_data = self.db['tar'][real_index]
                conv_data = torch.tensor(conv_data, dtype=torch.cfloat)
            elif self.mode == 'test':
                # In 'test' mode, load target if it exists, but it won't be returned by default.
                if self.has_target:
                    conv_data = torch.tensor(self.db['tar'][real_index], dtype=torch.cfloat)

            # ----- Patch extraction -----
            if self.patch_size is not None:
                C_dwi, D, H, W = dwi_data.shape # [Channels, Depth, Height, Width]

                # Ensure patch size is within volume dimensions
                if any(p > dim for p, dim in zip(self.patch_size, (D, H, W))):
                    raise ValueError(f"Patch size {self.patch_size} is larger than volume size ({D}, {H}, {W}).")

                z = random.randint(0, D - self.patch_size[0])
                y = random.randint(0, H - self.patch_size[1])
                x = random.randint(0, W - self.patch_size[2])

                dwi_data = dwi_data[:, z:z+self.patch_size[0], y:y+self.patch_size[1], x:x+self.patch_size[2]]
                if conv_data is not None:
                    conv_data = conv_data[:, z:z+self.patch_size[0], y:y+self.patch_size[1], x:x+self.patch_size[2]]

            # ----- Normalization -----
            dwi_data = self._normalize(dwi_data)
            if conv_data is not None:
                conv_data = self._normalize(conv_data)
            
            # Check for NaN or Inf values after normalization
            if not torch.isfinite(dwi_data).all():
                raise ValueError("Invalid dwi_data (NaN or Inf) after normalization")
            if conv_data is not None and not torch.isfinite(conv_data).all():
                raise ValueError("Invalid conv_data (NaN or Inf) after normalization")
            
            # ----- Transformations -----
            if self.transform:
                if conv_data is not None:
                    dwi_data, conv_data = self.apply_joint_transform(dwi_data, conv_data)
                else:
                    dwi_data = self.transform(dwi_data) # Assuming transform can handle single input (NOT YET)

            # ----- Return data -----
            if self.mode == 'train':
                return dwi_data, conv_data
            else:
                return dwi_data

        except Exception as e:
            # Optionally log or print the issue
            print(f"Skipping index {real_index} due to error: {e}")
            # Try next sample (wrap around if needed)
            new_index = (index + 1) % len(self.indices)
            return self.__getitem__(new_index)
        
    def _normalize(self, data):
        if self.normalization == 'max':
            max_mag = torch.amax(torch.abs(data), dim=(1, 2, 3), keepdim=True)
            return data / max_mag

        elif self.normalization == 'compand':
            data = data / torch.amax(torch.abs(data), dim=(1, 2, 3), keepdim=True)
            return mu_law_compress(data)

        elif self.normalization == 'mean_std':
            mean = torch.mean(data, dim=(1, 2, 3), keepdim=True)
            std = torch.std(data, dim=(1, 2, 3), keepdim=True) + 1e-6
            data = (data - mean) / std
            return data
        
        elif self.normalization == 'std':
            std = torch.std(data, dim=(1, 2, 3), keepdim=True) + 1e-6
            data /= std
            return data

        else:
            raise ValueError(f"Unknown normalization type: {self.normalization}")
        
    def apply_joint_transform(self, dwi_volume, conv_volume):
        # dwi_volume, conv_volume: torch.cfloat, shape [C, D, H, W]

        def complex_to_real_channels(x):
            # [C, D, H, W] complex → [2*C, D, H, W] real
            real = x.real
            imag = x.imag
            stacked = torch.stack([real, imag], dim=1)  # [C, 2, D, H, W]
            return stacked.view(-1, *x.shape[1:])  # [2*C, D, H, W]

        def real_channels_to_complex(x, C):
            # [2*C, D, H, W] real → [C, D, H, W] complex
            restored = x.view(C, 2, *x.shape[1:])  # [C, 2, D, H, W]
            real = restored[:, 0]
            imag = restored[:, 1]
            return torch.complex(real, imag)  # [C, D, H, W]

        # Convert both to real tensors
        dwi_real = complex_to_real_channels(dwi_volume)
        conv_real = complex_to_real_channels(conv_volume)

        # Create a TorchIO Subject
        subject = tio.Subject(
            dwi=tio.Image(tensor=dwi_real, type=tio.INTENSITY),
            conv=tio.Image(tensor=conv_real, type=tio.INTENSITY),
        )

        # Apply transform once
        transformed = self.transform(subject)

        # Convert back to complex
        C_dwi = dwi_volume.shape[0]
        C_conv = conv_volume.shape[0]
        dwi_transformed = real_channels_to_complex(transformed['dwi'].tensor, C_dwi)
        conv_transformed = real_channels_to_complex(transformed['conv'].tensor, C_conv)

        return dwi_transformed, conv_transformed
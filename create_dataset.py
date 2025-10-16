import os
import random
import argparse
from collections import defaultdict
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io
import gc
from tqdm import tqdm

# DEFINE FUNCTIONS ----------------------------------------------------
def extract_prefix(filename):
    for suffix in ['dwi.mat', 'tar.mat', 'conv.mat']:
        if filename.endswith(suffix):
            return filename[:-len(suffix)]
    return None

def select_random_paired_files(folder_paths, num_files_per_folder=None):
    selected_files = defaultdict(list)  # Dictionary to store matched pairs per folder

    for folder in folder_paths:
        if not os.path.exists(folder):
            print(f"Warning: Folder {folder} does not exist.")
            continue

        # Get all files in the folder
        mat_files = sorted(os.listdir(folder))

        # Dictionary to store matching dwi and conv files
        paired_files = {}

        for file in mat_files:
            prefix = extract_prefix(file)
            if prefix is None:
                continue

            if prefix not in paired_files:
                paired_files[prefix] = {'dwi': None, 'tar': None}

            if file.endswith('dwi.mat'):
                paired_files[prefix]['dwi'] = file
            elif file.endswith('tar.mat'):
                paired_files[prefix]['tar'] = file

        # Filter out incomplete pairs (ensure both dwi and conv exist)
        complete_pairs = [(v['dwi'], v['tar']) for v in paired_files.values() if v['dwi'] and v['tar']]
        if not complete_pairs:
            print(f"No valid pairs found in {folder}.")
            continue

        # Check if there are fewer pairs than requested
        if len(complete_pairs) < num_files_per_folder:
            print(f"Warning: Folder {folder} has only {len(complete_pairs)} valid pairs out of requested {num_files_per_folder}.")

        # Randomly select the specified number of pairs
        random.seed(42)
        if num_files_per_folder is not None:
            selected_pairs = random.sample(complete_pairs, min(num_files_per_folder, len(complete_pairs)))
        else:
            selected_pairs = complete_pairs

        # Store selected files with full paths
        selected_files[folder] = [(dwi, os.path.join(folder, dwi), tar, os.path.join(folder, tar)) for dwi, tar in selected_pairs]
        print(f"[{folder}] Selected {len(selected_pairs)} pairs out of {len(complete_pairs)}")


    return selected_files

def create_h5_from_paths(path_list, phase, output_dir):
    # Load one sample to infer shape
    sample_dwi = scipy.io.loadmat(path_list[0][0])['dwi_beamf'][()].astype(np.complex64)
    sample_tar = scipy.io.loadmat(path_list[0][1])['dwi_beamf_HR'][()].astype(np.complex64)
    
    dwi_shape = sample_dwi.shape  # (9, 192, 192, 192)
    tar_shape = (1,) + sample_tar.shape  # expand tar

    # Create HDF5 file
    out_path = os.path.join(output_dir, f"data_{phase}.h5")
    with h5py.File(out_path, "w") as h5f:
        dwi_ds = h5f.create_dataset(
            "dwi", shape=(len(path_list),) + dwi_shape,
            maxshape=(None,) + dwi_shape, dtype=np.complex64, compression="gzip"
        )
        tar_ds = h5f.create_dataset(
            "tar", shape=(len(path_list),) + tar_shape,
            maxshape=(None,) + tar_shape, dtype=np.complex64, compression="gzip"
        )


        valid_count = 0  # Keep track of how many valid samples were saved

        for i, (dwi_path, tar_path) in enumerate(tqdm(path_list, desc=f"Processing {phase} set")):
            try:
                dwi = scipy.io.loadmat(dwi_path)['dwi_beamf'][()].astype(np.complex64)
                tar = scipy.io.loadmat(tar_path)['dwi_beamf_HR'][()].astype(np.complex64)
                tar = np.expand_dims(tar, axis=0)

                if np.isnan(dwi).any() or np.isnan(tar).any():
                    print(f"NaNs detected in files:\n  DWI: {dwi_path}\n  TAR: {tar_path} — Skipping.")
                    continue

                assert dwi.shape == dwi_shape
                assert tar.shape == tar_shape

                dwi_ds[valid_count] = dwi
                tar_ds[valid_count] = tar
                valid_count += 1

            except Exception as e:
                print(f"Error at index {i} ({dwi_path}, {tar_path}): {e}")
                continue

            del dwi, tar
            gc.collect()

        dwi_ds.resize((valid_count,) + dwi_shape)
        tar_ds.resize((valid_count,) + tar_shape)
    print(f"Saved {valid_count} valid samples to {out_path}")
    return valid_count

# MAIN SCRIPT ----------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Create ultrasound dataset.")
    parser.add_argument("--num_files", type=int, default=None,
                        help="Number of file pairs per folder.")
    parser.add_argument("--output_dir", type=str, default="dataset",
                        help="Output directory for .h5 files.")
    args = parser.parse_args()

    folders = [
        'shape_models/heart_fg/beamform',  
        'shape_models/three_ellip/beamform',
        'shape_models/heart_mesh/beamform',
        'shape_models/mri/beamform',
        'shape_models/empty_ellipsoid/beamform',
        'shape_models/two_chambers/beamform'
    ]

    os.makedirs(args.output_dir, exist_ok=True)

    n_files = None if (args.num_files is None or args.num_files <= 0) else args.num_files
    selected_files = select_random_paired_files(folders, n_files)

    all_paths = [(dwi_path, tar_path)
                 for folder_pairs in selected_files.values()
                 for _, dwi_path, _, tar_path in folder_pairs]

    if not all_paths:
        print("No valid data found. Exiting.")
        return
    
    train_paths, test_paths = train_test_split(all_paths, test_size=0.1, random_state=42)

    valid_train_count = create_h5_from_paths(train_paths, phase="train", output_dir=args.output_dir)
    valid_test_count = create_h5_from_paths(test_paths, phase="test", output_dir=args.output_dir)

    print(f"\n Final dataset summary:")
    print(f"  Train samples: {valid_train_count}")
    print(f"  Test samples:  {valid_test_count}")
    print(f"  Saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
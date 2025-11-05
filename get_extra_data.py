import os
import urllib.request
import zipfile
import shutil
import requests

def get_zenodo_data():
    ZENODO_URL = "https://zenodo.org/records/17431793/files/scatterers_inp_data.zip"
    DOWNLOAD_DIR = "tmp_downloads"
    EXTRACT_DIR = "tmp_extract"
    PROJECT_ROOT = "shape_models"

    # map each folder name inside the zip to where it should end up
    DESTINATIONS = {
        "mri":          os.path.join(PROJECT_ROOT, "mri", "scatterers"),
        "fg":         os.path.join(PROJECT_ROOT, "fg", "scatterers"),
        "vtk_files":  os.path.join(PROJECT_ROOT, "heart_mesh", "vtk_files"),
    }

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    print("folders created")

    # Download the data
    zip_path = os.path.join(DOWNLOAD_DIR, "scatterers_inp_data.zip")
    print(f"Downloading dataset from {ZENODO_URL} ...")

    with requests.get(ZENODO_URL, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                f.write(chunk)
    print(f"Downloaded to {zip_path}")

    print("Extracting folders...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print(f"Extracted to {EXTRACT_DIR}")

    extracted_items = os.listdir(EXTRACT_DIR)
    if len(extracted_items) == 1 and os.path.isdir(os.path.join(EXTRACT_DIR, extracted_items[0])):
        root_dir = os.path.join(EXTRACT_DIR, extracted_items[0])
    else:
        root_dir = EXTRACT_DIR

    for folder_name, dest in DESTINATIONS.items():
        src_dir = os.path.join(root_dir, folder_name)
        if not os.path.exists(src_dir):
            print(f"Folder '{folder_name}' not found in archive, skipping.")
            continue

        os.makedirs(dest, exist_ok=True)
        print(f"Moving {folder_name} → {dest}")
        for item in os.listdir(src_dir):
            src_item = os.path.join(src_dir, item)
            dst_item = os.path.join(dest, item)
            shutil.move(src_item, dst_item)

    print("Data extraction complete.")

    shutil.rmtree(EXTRACT_DIR, ignore_errors=True)
    print("Cleaned up temporary files.")


if __name__ == "__main__":
    get_zenodo_data()


import os
import urllib.request
import zipfile
import shutil

def get_zenodo_data():
    ZENODO_URL = "https://zenodo.org/record/1234567/files/my_dataset.zip?download=1"
    DOWNLOAD_DIR = "/tmp_downloads"
    EXTRACT_DIR = "/tmp_extract"
    PROJECT_ROOT = "/shape_models"

    # map each folder name inside the zip to where it should end up
    DESTINATIONS = {
        "mri":          os.path.join(PROJECT_ROOT, "mri", "scatterers"),
        "fg":         os.path.join(PROJECT_ROOT, "fg", "scatterers"),
        "vtk_files":  os.path.join(PROJECT_ROOT, "heart_mesh", "vtk_files"),
    }

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)

    # Download the data
    zip_path = os.path.join(DOWNLOAD_DIR, "other_sources.zip")
    print(f"Downloading dataset from {ZENODO_URL} ...")

    urllib.request.urlretrieve(ZENODO_URL, zip_path)
    print(f"Downloaded to {zip_path}")

    print("Extracting folders...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print(f"Extracted to {EXTRACT_DIR}")

    for folder_name, dest in DESTINATIONS.items():
        src_dir = os.path.join(EXTRACT_DIR, folder_name)
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


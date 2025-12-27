import os 
import sys
import requests
import os
import tarfile
import h5py
from tqdm import tqdm
import argparse
import zipfile
from constants import DefaultPaths

def format_files(members, file_format):
    for tarinfo in members:
        if os.path.splitext(tarinfo.name.removesuffix(".gz"))[1] == "." + file_format:
            yield tarinfo

def download_zip(
    url,
    save_path, 
    fn, 
):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for any errors in the request

    zip_fn = f"{save_path}/{fn}.zip"

    # Open a local file for writing in binary mode.
    with open(zip_fn, "wb") as file:
        for chunk in tqdm(response.iter_content(chunk_size=8192)):
            if chunk:  # Filter out keep-alive new chunks.
                file.write(chunk)

    print(f"Downloaded, saved to {zip_fn}")
    with zipfile.ZipFile(zip_fn,"r") as zip_ref:
        zip_ref.extractall(f"{save_path}/{fn}")


def download_from_gcs_bucket(
    bucket_name,
    source_prefix,
    save_path,
):
    """Download from GCS using Python client"""
    os.makedirs(save_path, exist_ok=True)
    
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket(bucket_name)
    
    blobs = bucket.list_blobs(prefix=source_prefix)
    
    for blob in tqdm(blobs):
        file_path = os.path.join(save_path, blob.name.replace(source_prefix + "/", ""))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        blob.download_to_filename(file_path)
    
    print(f"Downloaded successfully to {save_path}")

        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download dataset by GEO ID')
    parser.add_argument('--save-path', type=str, 
                        default="/home/sxr280/Spatialformer/data_download/Spatial")
    parser.add_argument('--url', type=str, 
                        default=None)
    args = parser.parse_args()

    out_path = DefaultPaths.SPATIAL
    file_exist = os.path.exists(out_path)
    assert file_exist, "ERDANOTLOADED ERROR:,  Please mount ERDA FIRST"
    if not os.path.exists(f"{out_path}/raw"):
        os.mkdir(f"{out_path}/raw")
    
    fn = args.url.split("/")[-1].split(".")[0]

    download_zip(
        url=args.url,
        save_path=f"{out_path}/raw",
        fn=fn
    )

    
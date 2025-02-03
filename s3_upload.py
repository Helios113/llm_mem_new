import boto3
import os
import threading
from tqdm import tqdm


class ProgressPercentage(object):

    def __init__(self, filename, pbar):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self._pbar = pbar

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            self._pbar.update(bytes_amount)


def upload_folder_to_s3(
    folder_path, bucket_name, s3_endpoint_url="http://128.232.115.19:9000", s3_prefix=""
):
    s3_client = boto3.client("s3", endpoint_url=s3_endpoint_url)

    total_files = sum([len(files) for r, d, files in os.walk(folder_path)])
    uploaded_files = 0

    with tqdm(total=total_files, desc="Total Progress", unit="file") as total_pbar:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, folder_path)
                s3_key = os.path.join(s3_prefix, relative_path)
                with tqdm(total=os.path.getsize(file_path), unit='B', unit_scale=True, desc=file) as pbar:
                    s3_client.upload_file(
                        file_path, bucket_name, s3_key, Callback=ProgressPercentage(file_path, pbar)
                    )
                uploaded_files += 1
                total_pbar.update(1)


if __name__ == "__main__":
    s3_endpoint_url = "http://128.232.115.19:9000"
    s3_prefix = "data"
    bucket_name = "memorisation"

    s3_folder = "amazonqa"
    folder_path = (
        "/nfs-share/pa511/new_work/data/amazonqa"
    )
    print(s3_folder)
    upload_folder_to_s3(
        folder_path, bucket_name, s3_endpoint_url, os.path.join(s3_prefix, s3_folder)
    )

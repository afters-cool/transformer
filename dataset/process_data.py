import argparse
import json
import os
import re
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from dotenv import load_dotenv, main
from lxml import html
from tqdm import tqdm

TIME_SERIES_LENGTH = 672



def compute_loss(
    net: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device = "cpu",
) -> torch.Tensor:
    """Compute the loss of a network on a given dataset.
    Does not compute gradient.
    Parameters
    ----------
    net:
        Network to evaluate.
    dataloader:
        Iterator on the dataset.
    loss_function:
        Loss function to compute.
    device:
        Torch device, or :py:class:`str`.
    Returns
    -------
    Loss as a tensor with no grad.
    """
    running_loss = 0
    with torch.no_grad():
        for inp, out in dataloader:
            netout = net(inp.to(device)).cpu()
            running_loss += loss_function(out, netout)

    return running_loss / len(dataloader)


def download_from_url(session_requests, url, destination_folder):
    """
    @param: url to download file
    @param: dst place to put the file
    """
    response = session_requests.get(url, stream=True, headers=dict(referer=url))
    assert response.ok

    download_details = {}
    download_details["name"] = re.findall(
        "filename=(.+)", response.headers["content-disposition"]
    )[0]
    download_details["size"] = int(response.headers["Content-Length"])

    dst = destination_folder.joinpath(download_details["name"])
    if dst.is_file():
        first_byte = os.path.getsize(dst)
    else:
        first_byte = 0
    if first_byte >= download_details["size"]:
        return download_details["size"]
    header = {"Range": "bytes=%s-%s" % (first_byte, download_details["size"])}
    pbar = tqdm(
        total=download_details["size"],
        initial=first_byte,
        unit="B",
        unit_scale=True,
        desc=download_details["name"],
    )
    response = session_requests.get(url, headers=header, stream=True)
    assert response.ok

    with (open(dst, "ab")) as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
                pbar.update(1024)
    pbar.close()
    return download_details["size"]


class DownloadThread(threading.Thread):
    """
    DownloadThread
    """

    def __init__(self, session_requests, url, given_path):
        threading.Thread.__init__(self)
        self.session_requests = session_requests
        self.url = url
        self.path = given_path

    def run(self):
        """
        Download a file from given url
        """
        download_from_url(self.session_requests, self.url, self.path)


def npz_check(datasets_path, output_filename):
    """
    make sure npz is present
    """
    dataset_path = datasets_path.joinpath(output_filename + ".npz")
    x_train = {
        "url": "https://challengedata.ens.fr/participants/challenges/28/download/x-train",
        "filename": "x_train_LsAZgHU.csv",
    }
    y_train = {
        "url": "https://challengedata.ens.fr/participants/challenges/28/download/y-train",
        "filename": "y_train_EFo1WyE.csv",
    }
    x_test = {
        "url": "https://challengedata.ens.fr/participants/challenges/28/download/x-test",
        "filename": "x_test_QK7dVsy.csv",
    }
    make_npz_flag = False
    files_to_download = list()
    if not datasets_path.is_dir():
        # If there is no datasets folder
        os.makedirs(datasets_path)
        files_to_download = [x_train, y_train, x_test]
        make_npz_flag = True
    else:
        # If there is datasets folder
        if not datasets_path.joinpath(x_test["filename"]).is_file():
            # If there is datasets folder but there isn't x_test file
            files_to_download.append(x_test)
        if not dataset_path.is_file():
            # If there is datasets folder but there isn't output npz file
            make_npz_flag = True
            if not datasets_path.joinpath(x_train["filename"]).is_file():
                # If there is datasets folder but there isn't output npz file and there isn't
                # x_train file
                files_to_download.append(x_train)
            if not datasets_path.joinpath(y_train["filename"]).is_file():
                # If there is datasets folder but there isn't output npz file and there isn't
                # y_train file
                files_to_download.append(y_train)
    if files_to_download:
        login_url = "https://challengedata.ens.fr/login/"
        session_requests = requests.session()
        response = session_requests.get(login_url)
        assert response.ok

        authenticity_token = list(
            set(
                html.fromstring(response.text).xpath(
                    "//input[@name='csrfmiddlewaretoken']/@value"
                )
            )
        )[0]
        load_dotenv(".env.test.local")
        challenge_user_name = os.getenv("CHALLENGE_USER_NAME")
        challenge_user_password = os.getenv("CHALLENGE_USER_PASSWORD")

        if None in [challenge_user_name, challenge_user_password]:
            raise ValueError(
                "Missing login credentials. Make sure you follow https://github.com/DanielAtKrypton/ozechallenge_benchmark#dot-env-environment-variables"
            )
        response = session_requests.post(
            login_url,
            data={
                "username": challenge_user_name,
                "password": challenge_user_password,
                "csrfmiddlewaretoken": authenticity_token,
            },
            headers=dict(referer=login_url),
        )
        assert response.ok

        threads = []
        for file in files_to_download:
            threads.append(DownloadThread(session_requests, file["url"], datasets_path))

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

    if make_npz_flag:
        make_npz(
            datasets_path, output_filename, x_train["filename"], y_train["filename"]
        )
    return dataset_path


def make_npz(datasets_path, output_filename, x_train_filename, y_train_filename):
    """
    Creates the npz file and deletes the x_train and y_train files
    """
    x_train_path = datasets_path.joinpath(x_train_filename)
    y_train_path = datasets_path.joinpath(y_train_filename)
    print("Creating %s.npz..." % output_filename, end="\r")
    csv2npz(x_train_path, y_train_path, datasets_path, output_filename)
    clear_line_str = "\033[K"
    print(clear_line_str + "Create " + output_filename + ".npz\tDone")
    # there is no more need to keep x_train and y_train files
    # os.remove(x_train_path)
    # os.remove(y_train_path)
    print("Done")


def csv2npz(
    dataset_x_path, dataset_y_path, output_path, filename, labels_path="labels.json"
):
    """Load input dataset from csv and create x_train tensor."""
    # Load dataset as csv
    print("loading data")
    # 7501 rows
    x = pd.read_csv(dataset_x_path)
    y = pd.read_csv(dataset_y_path)

    print("loading labels.json")
    # Load labels, file can be found in challenge description
    with open(labels_path, "r") as stream_json:
        labels = json.load(stream_json)

    print("creating structures")
    m = x.shape[0]
    K = TIME_SERIES_LENGTH  # Can be found through csv

    # Create R and Z
    R = x[labels["R"]].values
    R = R.astype(np.float32)

    X = y[[f"{var_name}_{i}" for var_name in labels["X"] for i in range(K)]]
    X = X.values.reshape((m, -1, K))
    X = X.astype(np.float32)

    Z = x[[f"{var_name}_{i}" for var_name in labels["Z"] for i in range(K)]]
    Z = Z.values.reshape((m, -1, K))
    #     Z = Z.transpose((0, 2, 1))
    Z = Z.astype(np.float32)

    print("saving structures")
    np.savez(os.path.join(output_path, filename), R=R, X=X, Z=Z)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d",
        "--dataset_path",
        default="dataset/",
        type=Path,
        help="path to dataset",
    )
    parser.add_argument(
        "-x",
        "--dataset_x_path",
        default="x_train_LsAZgHU.csv",
        help="path to x train",
    )
    parser.add_argument(
        "-y",
        "--dataset_y_path",
        default="y_train_EFo1WyE.csv",
        help="path to y train",
    )
    parser.add_argument(
        "-o",
        "--output-filename",
        default="dataset",
        help="name of the npz file",
    )

    return parser


def main(args):

    parser = get_parser()
    args = parser.parse_args(args)

    datasets_path = args.dataset_path
    dataset_x_path = args.dataset_x_path
    dataset_y_path = args.dataset_y_path
    output_filename = args.output_filename

    make_npz(
        datasets_path=datasets_path,
        x_train_filename=dataset_x_path,
        y_train_filename=dataset_y_path,
        output_filename=output_filename,
    )


if __name__ == "__main__":

    # https://stackoverflow.com/a/14500228/5819113
    import sys

    sys.exit(main(sys.argv[1:]))
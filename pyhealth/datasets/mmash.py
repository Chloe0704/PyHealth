"""
Author: Meilin Liu
NetID: meilinl2
Paper Title: "PhysioMTL: Personalizing Physiological Patterns using Optimal Transport Multi-Task Regression"
Paper Link: https://arxiv.org/pdf/2203.12595

Description:
This file implements a dataset loader for the MMASH dataset, a publicly available
physiological dataset collected from 22 healthy subjects during daily life.
The loader downloads, extracts, and parses relevant tables (sleep, activity,
user info, questionnaire, RR intervals) into structured records compatible with
the PyHealth framework. It inherits from the BaseDataset class and supports
flexible table selection via the `tables` argument.
"""

import os
import pandas as pd
import zipfile
import requests
from tqdm import tqdm
from typing import List

from pyhealth.datasets import BaseDataset

MMASH_URL = "https://physionet.org/files/mmash/1.0.0/MMASH.zip"
MMASH_FOLDER = "MMASH"
ALL_TABLES = ["sleep", "activity", "user_info", "questionnaire", "rr"]


class MMASHDataset(BaseDataset):
    def __init__(
        self,
        root: str = "data",
        tables: List[str] = None,
        dev: bool = False,
        **kwargs
    ):
        if tables is None:
            raise ValueError(f"`tables` must be specified from {ALL_TABLES}")
        for table in tables:
            if table not in ALL_TABLES:
                raise ValueError(f"Unsupported table: {table}. Must be one of {ALL_TABLES}")

        self.tables = tables
        super().__init__(dataset_name="MMASH", root=root, tables=tables, dev=dev, **kwargs)

        self.dataset_folder = os.path.join(self.root, MMASH_FOLDER)
        if not os.path.exists(os.path.join(self.dataset_folder, "MMASH")):
            self.download_and_extract()

        self.load_data()

    def download_and_extract(self):
        os.makedirs(self.dataset_folder, exist_ok=True)
        zip_path = os.path.join(self.dataset_folder, "mmash.zip")

        response = requests.get(MMASH_URL, stream=True)
        total = int(response.headers.get("content-length", 0))
        with open(zip_path, "wb") as file, tqdm(
            desc="Downloading MMASH",
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.dataset_folder)
        os.remove(zip_path)

    def load_data(self):
        root_dir = os.path.join(self.dataset_folder, "MMASH")
        users = sorted(os.listdir(root_dir))
        for user_id in users:
            user_path = os.path.join(root_dir, user_id)
            if not os.path.isdir(user_path):
                continue

            record = {}
            try:
                if "sleep" in self.tables:
                    sleep = pd.read_csv(os.path.join(user_path, "sleep.csv"))
                    sleep["user_id"] = user_id
                    record["sleep"] = sleep
                if "activity" in self.tables:
                    activity = pd.read_csv(os.path.join(user_path, "Activity.csv"))
                    activity["user_id"] = user_id
                    record["activity"] = activity
                if "user_info" in self.tables:
                    info = pd.read_csv(os.path.join(user_path, "user_info.csv"))
                    info["user_id"] = user_id
                    record["user_info"] = info
                if "questionnaire" in self.tables:
                    quest = pd.read_csv(os.path.join(user_path, "questionnaire.csv"))
                    quest["user_id"] = user_id
                    record["questionnaire"] = quest
                if "rr" in self.tables:
                    rr = pd.read_csv(os.path.join(user_path, "RR.csv"))
                    rr["user_id"] = user_id
                    record["rr"] = rr

            except FileNotFoundError:
                continue

            self.add_sample(patient_id=user_id, visit_id=user_id, record=record)

# src/data/download.py
"""
Downloads ARC-AGI-1 dataset from Francois Chollet's GitHub repo.
Saves raw JSON files into data/raw/{split}/ directories.
"""

import json
import os
import requests

# Base URL for raw file access on GitHub
_GITHUB_RAW = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data"

# ARC-AGI-1 has three splits
SPLITS = ["training", "evaluation"]


def download_split(split: str, raw_dir: str) -> int:
    """
    Downloads all tasks for a given split.
    Returns the number of tasks downloaded.
    """
    assert split in SPLITS, f"Unknown split: {split}"

    split_dir = os.path.join(raw_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    # Fetch the directory listing via GitHub API (not raw)
    api_url = f"https://api.github.com/repos/fchollet/ARC-AGI/contents/data/{split}"
    print(f"[download] Fetching file list for split='{split}' ...")

    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    file_list = resp.json()

    # Filter to .json files only
    json_files = [f for f in file_list if f["name"].endswith(".json")]
    print(f"[download] Found {len(json_files)} tasks in '{split}'")

    downloaded = 0
    for file_info in json_files:
        task_id = file_info["name"]  # e.g. "007bbfb7.json"
        dest_path = os.path.join(split_dir, task_id)

        if os.path.exists(dest_path):
            downloaded += 1
            continue  # skip already downloaded

        raw_url = f"{_GITHUB_RAW}/{split}/{task_id}"
        r = requests.get(raw_url, timeout=30)
        r.raise_for_status()

        with open(dest_path, "w", encoding="utf-8") as f:
            json.dump(r.json(), f)

        downloaded += 1

    print(f"[download] '{split}': {downloaded}/{len(json_files)} tasks ready at {split_dir}")
    return downloaded


def download_all(raw_dir: str = "data/raw") -> None:
    """Downloads training, evaluation, and test splits."""
    for split in SPLITS:
        download_split(split, raw_dir)
    print("[download] All splits complete.")


if __name__ == "__main__":
    download_all()
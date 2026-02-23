import sys
import os
from pathlib import Path
import argparse

project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(1, project_root)

from src.data_processing.bio import create_BIO_tags

# Required console input args: train_path, dev_path, test_path, model_path, model_save_path

# Get inputs

# Put the inputs into AO experiments


# First, get the og data path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)

    parser.add_argument("--out", type=str, required=True)

    args = parser.parse_args()
    data_path = args.train
    out_dir   = args.out

    # Create and save tags
    create_BIO_tags(path=data_path, output_path=out_dir+'tagged.json', type='Opinion', train=True)
    create_BIO_tags(path=data_path, output_path=out_dir+'tagged.json', type='Aspect', train=True)
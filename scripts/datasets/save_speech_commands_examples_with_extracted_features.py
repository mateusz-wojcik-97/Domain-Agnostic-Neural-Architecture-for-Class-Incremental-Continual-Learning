import warnings
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from pyannote.audio import Model, Inference
from tqdm import tqdm

from settings import DATASETS_DIR

DATASET_NAME = 'speechcommands_pyannote_512'
BASE_PATH = Path('Your path to the dataset')
CLASSES_TO_EXPORT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


if __name__ == '__main__':

    warnings.filterwarnings("ignore")  # ".*does not have many workers.*"

    model = Model.from_pretrained(
        "pyannote/embedding", use_auth_token="your token"
    )

    inference = Inference(model, window="whole")

    count_failed = 0
    index = 0
    for subset_name in ['train', 'test']:
        X, y = [], []

        logging_dir = DATASETS_DIR / 'ensemble_e2e' / DATASET_NAME
        print(f"Saving dataset into folder: {logging_dir}")

        dataset = load_dataset("speech_commands", name="v0.01", split=subset_name)
        print(len(dataset))
        for item in tqdm(dataset):
            try:
                label = item['label']

                if label in CLASSES_TO_EXPORT:
                    path = BASE_PATH / item['audio']['path']
                    embedding = inference(path)
                    X.append(embedding)
                    y.append(label)

                    current_class_dir = logging_dir / subset_name / str(label)
                    current_class_dir.mkdir(parents=True, exist_ok=True)

                    X_tensor = torch.from_numpy(embedding.flatten())
                    torch.save(X_tensor, current_class_dir / f"{index}.pt")
                    index += 1
            except Exception as exp:
                count_failed += 1
                print(exp)

        print(f"{subset_name}: {np.shape(X)} {np.shape(y)}")
        print(f"Failed: {count_failed}")

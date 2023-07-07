from pathlib import Path
from typing import Tuple, List, Optional

import torch
from avalanche.benchmarks.datasets import DatasetFolder
from avalanche.benchmarks.utils.dataset_definitions import IDatasetWithTargets
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from continual_learning.models.feature_extractors.base import FeatureExtractor
from settings import STORAGE_DIR

BBC_NEWS_DATASET_DIR = STORAGE_DIR / 'datasets' / 'raw' / 'bbc_news'


class BBCNewsDataset(Dataset, IDatasetWithTargets):
    def __init__(self, texts, labels, target_names: Optional[List[str]] = None):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels
        self.targets = list(labels)
        self.target_names = target_names

    def __getitem__(self, index) -> Tuple[str, torch.Tensor]:
        return self.texts[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)


def load_to_string(path: Path) -> str:
    with open(path, 'r') as file:
        return file.read()


def get_bbc_news_dataset(
    subset: str = "train",
    feature_extractor: Optional[FeatureExtractor] = None,
    limit: Optional[int] = None,
) -> BBCNewsDataset:
    dataset = DatasetFolder(
        root=BBC_NEWS_DATASET_DIR,
        loader=load_to_string,
        extensions=['.txt'],
    )

    texts = []
    labels = []
    target_names = dataset.classes
    for text, label in dataset:
        texts.append(text)
        labels.append(label)

    split = train_test_split(
        texts, labels, test_size=0.25, random_state=10, shuffle=True
    )
    if subset == "train":
        texts, _, labels, _ = split
    elif subset == "test":
        _, texts, _, labels = split
    else:
        raise ValueError(f"Incorrect subset passed {subset}")

    if limit:
        texts, labels = texts[:limit], labels[:limit]

    print(f"Dataset size (examples): {len(labels)}")

    if feature_extractor:
        print("Extracting features...")
        texts = [feature_extractor.get_features(text) for text in tqdm(texts)]

        print(texts[0].shape)

    print(target_names)
    return BBCNewsDataset(texts, labels, target_names)


if __name__ == '__main__':
    dataset = get_bbc_news_dataset(subset="train")
    dataset = get_bbc_news_dataset(subset="test")
    print(dataset)

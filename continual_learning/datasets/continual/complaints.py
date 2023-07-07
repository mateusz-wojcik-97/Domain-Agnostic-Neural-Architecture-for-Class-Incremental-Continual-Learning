from collections import defaultdict
from typing import Tuple, List, Optional

import torch
from avalanche.benchmarks.utils.dataset_definitions import IDatasetWithTargets
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from continual_learning.models.feature_extractors.base import FeatureExtractor


class ComplaintsDataset(Dataset, IDatasetWithTargets):
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


def get_complaints_dataset(
    subset: str = "train",
    feature_extractor: Optional[FeatureExtractor] = None,
    limit_per_class: int = 2000,
    num_classes: int = 10,
) -> ComplaintsDataset:
    dataset = load_dataset("consumer-finance-complaints", ignore_verifications=True)
    train = dataset["train"]

    dataset_stats = defaultdict(int)

    texts = []
    labels = []
    for index, item in enumerate(tqdm(train, total=len(train))):
        label = item['Product']
        complaint_text = item['Complaint Text']

        if dataset_stats[label] < limit_per_class and complaint_text:
            text = item['Issue'] + ' ' + item['Sub Issue'] + ' ' + complaint_text
            texts.append(text)
            labels.append(label)

            dataset_stats[label] += 1

    selected_labels = {k: v for index, (k, v) in enumerate(sorted(dataset_stats.items(), key=lambda item: item[1], reverse=True)) if index < num_classes}
    selected_labels = {key: value for key, value in selected_labels.items() if value >= limit_per_class}
    texts = [text for text, label in zip(texts, labels) if label in selected_labels]
    labels = [label for label in labels if label in selected_labels]

    assert len(texts) == len(labels)

    print(f"Dataset stats: {dataset_stats}")

    import numpy as np
    target_names, counts = np.unique(labels, return_counts=True)
    target_names = target_names.tolist()
    labels = [target_names.index(label) for label in labels]

    print(f'Targets: {target_names} {len(target_names)} {counts}')
    print(f"Dataset size (examples): {len(labels)}")

    split = train_test_split(
        texts, labels, test_size=0.2, random_state=10, shuffle=True
    )
    if subset == "train":
        texts, _, labels, _ = split
    elif subset == "test":
        _, texts, _, labels = split
    else:
        raise ValueError(f"Incorrect subset passed {subset}")

    if feature_extractor:
        print("Extracting features...")
        texts = [feature_extractor.get_features(text) for text in tqdm(texts)]

        print(texts[0].shape)

    return ComplaintsDataset(texts, labels, target_names)


if __name__ == '__main__':
    datasets = get_complaints_dataset(subset="train", limit_per_class=2000, num_classes=10)
    datasets = get_complaints_dataset(subset="test", limit_per_class=2000, num_classes=10)

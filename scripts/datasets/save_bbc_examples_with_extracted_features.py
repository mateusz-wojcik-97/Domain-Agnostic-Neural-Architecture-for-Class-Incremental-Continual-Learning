import warnings
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from continual_learning.dataset.continual.bbc_news import get_bbc_news_dataset
from continual_learning.models.feature_extractors.transformer_feature_extractor import \
    TransformerFeatureExtractor
from settings import CONFIGS_DIR, DATASETS_DIR

DATASET_NAME = 'bbcnews_distilbert_512'


def save_dataset(dataset: Any, folder: str) -> None:
    index = 0
    data_loader = DataLoader(dataset=dataset, batch_size=1)
    for example in tqdm(data_loader, total=len(data_loader)):
        current_class = example[1].item()  # Assuming that this is CI scenario

        current_class_dir = logging_dir / folder / str(current_class)
        current_class_dir.mkdir(parents=True, exist_ok=True)

        torch.save(example[0].flatten(), current_class_dir / f"{index}.pt")
        index += 1


if __name__ == '__main__':
    warnings.filterwarnings("ignore")  # ".*does not have many workers.*"
    configs = OmegaConf.load(CONFIGS_DIR / 'ensemble_e2e' / 'datasets_config.yml')
    config = configs[DATASET_NAME]

    for subset_name in ['train', 'test']:
        feature_extractor = TransformerFeatureExtractor(
            model_name=config.model_name,
            tokenizer_name=config.tokenizer_name,
            max_sentence_len=config.max_sentence_len,
        )
        dataset = get_bbc_news_dataset(
            subset=subset_name,
            feature_extractor=feature_extractor,
            limit=config.limit
        )

        logging_dir = DATASETS_DIR / 'ensemble_e2e' / DATASET_NAME
        print(f"Saving dataset into folder: {logging_dir}")

        save_dataset(dataset=dataset, folder=subset_name)

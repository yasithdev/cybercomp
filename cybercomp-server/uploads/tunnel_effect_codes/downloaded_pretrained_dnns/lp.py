from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from linear_probe.feature_extractor import Feature_Extractor
from linear_probe.linear_probor import Linear_Probe

parser = ArgumentParser()
parser.add_argument("--model_name", default="resnet50", type=str, help="Model to load")
parser.add_argument("--dataset_name", default="imagenet", type=str, help="Dataset name")
parser.add_argument("--data_dir", default="/data/datasets/ImageNet-100", type=str, help="Data directory")
parser.add_argument("--features_per_file", default=10000, type=int, help="Number of features per file")
parser.add_argument(
    "--num_classes_pretrained", default=1000, type=int, help="Number of classes in pretrained model"
)
parser.add_argument("--batch_size", default=512, type=int, help="Batch size")
parser.add_argument("--feature_root", default="./features/", type=str)

parser.add_argument("--epochs", default=30, type=int, help="Number of epochs")
parser.add_argument("--lr", default=5e-4, type=float, help="Learning rate")

parser.add_argument("--num_classes", default=100, type=int, help="Number of classes")

parser.add_argument("--num_workers", default=8, type=int, help="Number of workers for dataloader")
parser.add_argument("--wd", default=0.0, type=float, help="Weight decay")


if __name__ == "__main__":
    cfg = parser.parse_args()
    Feature_Extractor(cfg).extract()

    exp_path = Path(f"lt_results_{cfg.lr}")
    exp_path = exp_path / cfg.model_name / cfg.dataset_name
    exp_path.mkdir(parents=True, exist_ok=True)
    cfg.exp_path = exp_path
    lp_acc = Linear_Probe(cfg).probe()
    print(lp_acc)

    file_path = exp_path / "acc.npy"

    np.save(file_path, lp_acc)

from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset

from models.models import get_names_of_layers


# def get_feature_files(feature_dir: str, split: str):
def get_feature_files(model_name, dataset_name, split, layer_names):
    """Get layerwise feature files (A list of features for each file)

    Args:
        feature_dir (str): _description_
        split (str): _description_

    Returns:
        dict {str:list}. LayerID is the key and value is a list of feature files for that layer
    """
    feature_file_dirs = Path("features") / model_name / split / dataset_name

    out = {}
    layer = 0
    for layer_name in layer_names:
        out[layer] = feature_file_dirs / layer_name
        layer += 1

    return out


class Features(Dataset):
    # def __init__(
    #     self,
    #     feature_dir: str = "features/Flowers_-0.01_20_l3/gen9",
    #     split: str = "train",
    #     all_in_ram: bool = True,
    # ):
    def __init__(
        self,
        model_name,
        dataset_name,
        split,
    ):
        all_in_ram: bool = True
        super(Features, self).__init__()
        assert split in ["train", "val", "test"], f"split should be train/val/test but given {split}"
        assert all_in_ram == True, "Current implementation assumes we have enough RAM for all layers"
        self.layer_names = get_names_of_layers(model_name)
        self.feature_dir = get_feature_files(model_name, dataset_name, split, self.layer_names)
        self.layers = sorted(list(self.feature_dir.keys()))
        print(self.layers)
        print(len(self.layers))
        self.set_layer_called = False

    def set_layer(self, idx):
        self.set_layer_called = True

        # self.current_layer = self.layers[idx]
        # feature_files = self.feature_dir[self.current_layer]
        feature_file_dir = self.feature_dir[idx]
        layer_name = self.layer_names[idx]

        self.features = []
        self.targets = []
        files = sorted(list(feature_file_dir.glob("*.pt")))
        for file in files:

            # for file in feature_files:
            # https://github.com/pytorch/pytorch/issues/40403
            torch_data = torch.load(file, map_location=torch.device("cpu"))
            features = torch_data["features"]
            targets = torch_data["targets"]
            self.features.extend(features)
            self.targets.extend(targets)
        print(f"Shape of features: {len(self.features)}")
        return layer_name, features.shape[1]

    def len_layers(self):
        print(f"layers are\n {[f'{idx}:{l}' for idx, l in enumerate(self.layers)]}")
        print("\nNumber of layers:", len(self.layers))
        return len(self.layers)

    def __getitem__(self, idx):
        assert self.set_layer_called, "User must call set_layer before getting items"
        feature = self.features[idx]
        target = self.targets[idx]
        return feature, target

    def __len__(self):
        return len(self.features)


def load_features(cfg):
    trainset = Features(cfg.model_name, cfg.dataset_name, split="train")
    valset = Features(cfg.model_name, cfg.dataset_name, split="val")

    # testset = Features(feature_dir, split="test")

    trainloader = DataLoader(
        trainset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True
    )
    valloader = DataLoader(valset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    testloader = None

    # if cfg.eval_tst:
    #     testloader = DataLoader(
    #         testset, batch_size=cfg.knn_test_batch_size, num_workers=cfg.knn_num_workers, pin_memory=True
    #     )
    #     if cfg.knn_test_only:
    #         valloader = None

    return trainloader, valloader, testloader


if __name__ == "__main__":
    cfg = SimpleNamespace(
        knn_all_in_ram=1,
        knn_num_workers=6,
        eval_tst=1,
        knn_test_only=1,
        knn_train_batch_size=2,
        knn_test_batch_size=2,
    )
    # dataset = Features(feature_dir='features/Flowers_-0.01_20_l3/gen9', split='train')
    tl, vl, tesl = load_features(cfg, "features/Flowers_-0.01_20_l3/gen9/best_val.pt")
    tl.dataset.set_layer(2)
    tesl.dataset.set_layer(2)
    print(tesl)

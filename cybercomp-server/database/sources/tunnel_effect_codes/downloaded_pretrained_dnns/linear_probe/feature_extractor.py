from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from timm.models.vision_transformer import Block
from torchvision.models.vision_transformer import EncoderBlock
from tqdm.contrib import tqdm

from dataloader.data_utils import get_img_loaders
from models.models import feature_processor, get_ssl_model


class Feature_Extractor:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        # self.dataset_name = cfg.set
        self.dataset_name = cfg.dataset_name
        self.feature_dir = Path(cfg.feature_root, cfg.model_name)
        print(f"Saving features to {self.feature_dir}")
        model_name = cfg.model_name

        model = get_ssl_model(model_name)
        self.layer_feature_processor = feature_processor(model_name)
        if "resnet" in model_name:
            self.target_block = nn.Conv2d

        elif "convnext" in model_name:
            self.target_block = nn.Conv2d
        elif "vit" in model_name:
            self.target_block = EncoderBlock
        else:
            self.target_block = Block
        self.model = model
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.model.cuda().eval()
        self.hooks = self.attach_hooks()
        self.keys = list(self.hooks.keys())
        print(f"Extracting features from {self.keys} layers")

        train_loader, val_loader = get_img_loaders(cfg.data_dir, resolution=224, batch_size=cfg.batch_size)
        self.loaders = {"train": train_loader, "val": val_loader}

        self.activations = {}

    def attach_hooks(self):
        hooks = {}

        def _getActivation(name):
            # the hook signature
            def hook(model, input, output):
                self.activations[name.replace(".", "")] = self.layer_feature_processor(output)

            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, self.target_block):
                if "downsample" in name or "downsampling" in name:
                    continue
                hooks[name.replace(".", "")] = module.register_forward_hook(_getActivation(name))

        return hooks

    def _dummy_batch(self):
        print("Forwarding a dummy batch")
        x, _ = next(iter(self.loaders["train"]))
        y = self.model(x.cuda())
        # self.activations["out"] = y
        pool = nn.AdaptiveAvgPool2d(1)
        # for k, v in self.activations.items():
        #     print(f"After layer {k}, shape was {v.shape}")
        #     print(type(v))

    def check_features_exist(self, mode):
        self._dummy_batch()

        feature_dir = self.feature_dir / mode
        print(f"Checking if features exist in {feature_dir}")
        # feature_files = list(feature_dir.glob("*.pt"))

        # check which features exist. This line drops the _POSTFIX
        # feature_files = ["_".join(f.stem.split("_")[:-1]) for f in feature_files]
        # feature_dir.mkdir(parents=True, exist_ok=True)

        for k in self.activations:
            temp_dir = feature_dir / self.dataset_name / k
            files = list(temp_dir.glob("*.pt"))
            # if f"{k}_{self.dataset_name}" not in feature_files:
            # Something didn't exist
            if len(files) == 0:
                return False, None

        return True, feature_dir

    def save_features(self, features, targets, mode, postfix):
        feature_dir = self.feature_dir / mode
        save_dir = feature_dir / self.dataset_name
        dict_to_save = {}
        for k in features:
            dict_to_save[k] = {"features": features[k], "targets": targets[k]}
            temp_save_dir = save_dir / k
            temp_save_dir.mkdir(parents=True, exist_ok=True)
            feature_file = temp_save_dir / f"{postfix}.pt"
            torch.save(dict_to_save[k], feature_file)
            print(f"saved at {feature_file}")

        dict_to_save = {}

    @torch.no_grad()
    def _extract(self, loader, mode="train", progressbar=True):
        # If already extracted, don't extract
        already_extracted, _fd = self.check_features_exist(mode)
        if already_extracted:
            print(f"All files are alreay extracted! To force feature_extraction, remove all features")
            return

        # dicts to store features and targets for saving on disk
        features = {}
        targets = {}
        total_processed = 0

        with tqdm(loader, disable=not progressbar) as t:
            for batch_idx, (inputs, _targets) in enumerate(t):  # TODO wrap it with tqdm
                total_processed += inputs.size(0)
                self.activations = {}
                inputs, _targets = inputs.cuda(), _targets.long().squeeze().cuda()
                out = self.model(inputs)
                # Get output of the network as a feature
                # self.activations["out"] = out

                # Append features and targets to the dictionaries
                for k in self.activations:
                    if k not in features:
                        features[k] = self.activations[k]
                        targets[k] = _targets
                    else:
                        features[k] = torch.cat([features[k], self.activations[k]], dim=0)
                        targets[k] = torch.cat([targets[k], _targets], dim=0)

                # Log progress in TQDM
                t.set_postfix(progress=f"{len(features[k])}/{len(loader.dataset)}")

                # Save features periodically to avoid OOM
                if len(features[k]) + self.cfg.batch_size > self.cfg.features_per_file:
                    self.save_features(features, targets, mode, total_processed)
                    features = {}
                    targets = {}

        if len(features) > 0:  # the leftovers
            self.save_features(features, targets, mode, total_processed)
            features = {}
            targets = {}

    def extract(self, progressbar=True):
        modes = ["train", "val"]

        for m in modes:
            print(f"extracting features for {m}")
            loader = self.loaders[m]
            self._extract(loader, m, progressbar)

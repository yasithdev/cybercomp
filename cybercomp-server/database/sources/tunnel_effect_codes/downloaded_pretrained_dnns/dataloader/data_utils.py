import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from timm.models.vision_transformer import Block
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from dataloader.cub import Cub2011


def get_img_loaders(data_path, resolution, batch_size=512, num_workers=8, pin_memory=True):
    batch_size = 256

    if "ImageNet-100" == data_path.split("/")[-1]:
        size = int((256 / 224) * resolution)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_noaug = transforms.Compose(
            [
                transforms.Resize(size, interpolation=3),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                normalize,
            ]
        )
        train_dataset_img = datasets.ImageFolder(os.path.join(data_path, "train"), transform_noaug)
        val_dataset_img = datasets.ImageFolder(os.path.join(data_path, "val"), transform_noaug)
        train_loader = torch.utils.data.DataLoader(
            train_dataset_img,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset_img,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    elif "imagenet-r" == data_path.split("/")[-1]:
        # path=/home/yousuf/dualprompt-pytorch/data/imagenet-r
        transform_noaug = get_imagenet_noaug_transforms(resolution)
        train_dataset_img = datasets.ImageFolder(os.path.join(data_path, "train"), transform_noaug)
        val_dataset_img = datasets.ImageFolder(os.path.join(data_path, "test"), transform_noaug)
        train_loader = torch.utils.data.DataLoader(
            train_dataset_img,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset_img,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    elif "ImageNet2012" == data_path.split("/")[-1]:
        transform_noaug = get_imagenet_noaug_transforms(resolution)
        train_dataset = datasets.ImageFolder(os.path.join(data_path, "train"), transform_noaug)
        val_dataset = datasets.ImageFolder(os.path.join(data_path, "val"), transform_noaug)

        # only 50 samples for each class
        train_idx = []
        for i in range(1000):
            idx = np.where(np.array(train_dataset.targets) == i)[0][:50]
            train_idx.extend(idx)

        train_dataset = Subset(train_dataset, train_idx)
        print(f"Using {len(train_idx)} samples for training")

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    elif "ImageNet-R" == data_path.split("/")[-1]:
        size = int((256 / 224) * resolution)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_noaug = transforms.Compose(
            [
                transforms.Resize(size, interpolation=3),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                normalize,
            ]
        )
        train_dataset_img = datasets.ImageFolder(os.path.join(data_path, "train"), transform_noaug)
        val_dataset_img = datasets.ImageFolder(os.path.join(data_path, "val"), transform_noaug)
        train_loader = torch.utils.data.DataLoader(
            train_dataset_img,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset_img,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    elif "ImageNet-10" == data_path.split("/")[-1]:
        size = int((256 / 224) * resolution)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_noaug = transforms.Compose(
            [
                transforms.Resize(size, interpolation=3),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                normalize,
            ]
        )
        train_dataset_img = datasets.ImageFolder(os.path.join(data_path, "train"), transform_noaug)
        val_dataset_img = datasets.ImageFolder(os.path.join(data_path, "val"), transform_noaug)
        train_loader = torch.utils.data.DataLoader(
            train_dataset_img,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset_img,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    elif "NINCO" in data_path:
        size = int((256 / 224) * resolution)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_noaug = transforms.Compose(
            [
                transforms.Resize(size, interpolation=3),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                normalize,
            ]
        )
        dataset_img = datasets.ImageFolder(data_path, transform_noaug)
        train_idx, valid_idx = train_test_split(
            np.arange(len(dataset_img)), test_size=0.2, random_state=42, stratify=dataset_img.targets
        )
        train_dataset_img = Subset(dataset_img, train_idx)
        val_dataset_img = Subset(dataset_img, valid_idx)
        train_loader = torch.utils.data.DataLoader(
            train_dataset_img,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset_img,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    elif "CIFAR10" == data_path.split("/")[-1]:
        # https://github.com/aimagelab/mammoth/blob/master/datasets/seq_cifar10.py
        size = int((256 / 224) * resolution)
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2615])
        transform_noaug = transforms.Compose(
            [
                transforms.Resize(size, interpolation=3),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                normalize,
            ]
        )

        train_dataset = datasets.CIFAR10(data_path, train=True, transform=transform_noaug, download=True)
        val_dataset = datasets.CIFAR10(data_path, train=False, transform=transform_noaug, download=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )

    elif "CIFAR100" == data_path.split("/")[-1]:
        size = int((256 / 224) * resolution)
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        transform_noaug = transforms.Compose(
            [
                transforms.Resize(size, interpolation=3),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                normalize,
            ]
        )

        train_dataset = datasets.CIFAR100(data_path, train=True, transform=transform_noaug, download=True)
        val_dataset = datasets.CIFAR100(data_path, train=False, transform=transform_noaug, download=True)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )

    elif "Flowers102" == data_path.split("/")[-1]:
        # 102
        transform_noaug = get_imagenet_noaug_transforms(resolution)

        train_dataset = datasets.Flowers102(
            data_path, split="train", transform=transform_noaug, download=True
        )
        val_dataset = datasets.Flowers102(data_path, split="val", transform=transform_noaug, download=True)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
    elif "CUB" == data_path.split("/")[-1]:
        transform_noaug = get_imagenet_noaug_transforms(resolution)
        train_dataset = Cub2011(data_path, train=True, transform=transform_noaug, download=True)
        val_dataset = Cub2011(data_path, train=False, transform=transform_noaug, download=True)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )

    elif "FGVCAircraft" == data_path.split("/")[-1]:
        # 102
        transform_noaug = get_imagenet_noaug_transforms(resolution)
        train_dataset = datasets.FGVCAircraft(
            data_path, split="train", transform=transform_noaug, download=True
        )
        val_dataset = datasets.FGVCAircraft(data_path, split="val", transform=transform_noaug, download=True)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )

    elif "Pets37" == data_path.split("/")[-1]:
        # 37
        transform_noaug = get_imagenet_noaug_transforms(resolution)

        train_dataset = datasets.OxfordIIITPet(
            data_path, split="trainval", transform=transform_noaug, download=True
        )
        val_dataset = datasets.OxfordIIITPet(
            data_path, split="test", transform=transform_noaug, download=True
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )

    elif "SLT10" == data_path.split("/")[-1]:
        # 10
        # https://github.com/YU1ut/MixMatch-pytorch/pull/25/files#diff-41a58fae86b8dd31bf58e9441163057e115da9aa493af90d1e188f511602527eR16
        size = int((256 / 224) * resolution)
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
        transform_noaug = transforms.Compose(
            [
                transforms.Resize(size, interpolation=3),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                normalize,
            ]
        )

        train_dataset = datasets.STL10(data_path, split="train", transform=transform_noaug, download=True)
        val_dataset = datasets.STL10(data_path, split="test", transform=transform_noaug, download=True)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )

    elif "SVHN" == data_path.split("/")[-1]:
        # 10
        size = int((256 / 224) * resolution)
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2615])
        transform_noaug = transforms.Compose(
            [
                transforms.Resize(size, interpolation=3),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                normalize,
            ]
        )

        train_dataset = datasets.SVHN(data_path, split="train", transform=transform_noaug, download=True)
        val_dataset = datasets.SVHN(data_path, split="test", transform=transform_noaug, download=True)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )

    elif "Places" == data_path.split("/")[-3]:
        transform_noaug = get_imagenet_noaug_transforms(resolution)

        train_dataset = datasets.ImageFolder(os.path.join(data_path, "train"), transform_noaug)
        val_dataset = datasets.ImageFolder(os.path.join(data_path, "val"), transform_noaug)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    print("Train loader length: ", len(train_loader.dataset))
    print("Val loader length: ", len(val_loader.dataset))

    return train_loader, val_loader


def get_imagenet_noaug_transforms(resolution):
    size = int((256 / 224) * resolution)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_noaug = transforms.Compose(
        [
            transforms.Resize(size, interpolation=3),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return transform_noaug


def get_feature_loaders(feature_extractor, train_loader_img, val_loader_img):
    print("getting features...")
    training_features = []
    training_targets = []
    for data, target in tqdm(train_loader_img):
        data, target = data.to("cuda"), target.to("cuda")
        with torch.no_grad():
            features = feature_extractor(data)
            features = torch.flatten(features, 1).detach().cpu()
        training_features.append(features)
        training_targets.append(target.detach().cpu())
    training_features = torch.cat(training_features, dim=0)
    training_targets = torch.cat(training_targets, dim=0)

    val_features = []
    val_targets = []
    for data, target in tqdm(val_loader_img):
        data, target = data.to("cuda"), target.to("cuda")
        with torch.no_grad():
            features = feature_extractor(data)
            features = torch.flatten(features, 1).detach().cpu()
        val_features.append(features)
        val_targets.append(target.detach().cpu())
    val_features = torch.cat(val_features, dim=0)
    val_targets = torch.cat(val_targets, dim=0)

    return training_features, training_targets, val_features, val_targets


def get_dict_map_layers(model_name, model):
    # Key layer depth: layer index inside network for conv2d"
    dict_map_layers_all = {
        "VGG13": {1: 1, 2: 4, 3: 8, 4: 11, 5: 15, 6: 18, 7: 22, 8: 25, 9: 29, 10: 32},
        "VGG13small": {1: 1, 2: 4, 3: 8, 4: 11, 5: 15, 6: 18, 7: 22, 8: 25, 9: 29, 10: 32},
        "VGG19": {
            1: 1,
            2: 4,
            3: 8,
            4: 11,
            5: 15,
            6: 18,
            7: 21,
            8: 24,
            9: 28,
            10: 31,
            11: 34,
            12: 37,
            13: 41,
            14: 44,
            15: 47,
            16: 50,
        },
    }
    if model_name in dict_map_layers_all:
        return dict_map_layers_all[model_name]
    elif model_name in ["dino", "mugs", "mae"]:
        dict_map_layers = {}
        layer_idx = 1
        target_layer_idx = 1
        for name, module in model.named_modules():
            if isinstance(module, Block):
                dict_map_layers[layer_idx] = target_layer_idx
                layer_idx += 1
            target_layer_idx += 1
        return dict_map_layers


imagenet100_class = [
    "rocking_chair",
    "pirate",
    "computer_keyboard",
    "Rottweiler",
    "Great_Dane",
    "tile_roof",
    "harmonica",
    "langur",
    "Gila_monster",
    "hognose_snake",
    "vacuum",
    "Doberman",
    "laptop",
    "gasmask",
    "mixing_bowl",
    "robin",
    "throne",
    "chime",
    "bonnet",
    "komondor",
    "jean",
    "moped",
    "tub",
    "rotisserie",
    "African_hunting_dog",
    "kuvasz",
    "stretcher",
    "garden_spider",
    "theater_curtain",
    "honeycomb",
    "garter_snake",
    "wild_boar",
    "pedestal",
    "bassinet",
    "pickup",
    "American_lobster",
    "sarong",
    "mousetrap",
    "coyote",
    "hard_disc",
    "chocolate_sauce",
    "slide_rule",
    "wing",
    "cauliflower",
    "American_Staffordshire_terrier",
    "meerkat",
    "Chihuahua",
    "lorikeet",
    "bannister",
    "tripod",
    "head_cabbage",
    "stinkhorn",
    "rock_crab",
    "papillon",
    "park_bench",
    "reel",
    "toy_terrier",
    "obelisk",
    "walking_stick",
    "cocktail_shaker",
    "standard_poodle",
    "cinema",
    "carbonara",
    "red_fox",
    "little_blue_heron",
    "gyromitra",
    "Dutch_oven",
    "hare",
    "dung_beetle",
    "iron",
    "bottlecap",
    "lampshade",
    "mortarboard",
    "purse",
    "boathouse",
    "ambulance",
    "milk_can",
    "Mexican_hairless",
    "goose",
    "boxer",
    "gibbon",
    "football_helmet",
    "car_wheel",
    "Shih-Tzu",
    "Saluki",
    "window_screen",
    "English_foxhound",
    "American_coot",
    "Walker_hound",
    "modem",
    "vizsla",
    "green_mamba",
    "pineapple",
    "safety_pin",
    "borzoi",
    "tabby",
    "fiddler_crab",
    "leafhopper",
    "Chesapeake_Bay_retriever",
    "ski_mask",
]

imagenet_r_class = [
    "pickup",
    "Pomeranian",
    "Saint_Bernard",
    "bagel",
    "lion",
    "cellular_telephone",
    "sax",
    "lemon",
    "strawberry",
    "Shih-Tzu",
    "standard_poodle",
    "bloodhound",
    "Siberian_husky",
    "Boston_bull",
    "tarantula",
    "burrito",
    "pug",
    "toy_poodle",
    "ostrich",
    "fox_squirrel",
    "porcupine",
    "espresso",
    "harmonica",
    "warplane",
    "lorikeet",
    "grasshopper",
    "bell_pepper",
    "grey_whale",
    "broccoli",
    "candle",
    "peacock",
    "starfish",
    "golden_retriever",
    "head_cabbage",
    "Afghan_hound",
    "ladybug",
    "parachute",
    "hippopotamus",
    "vase",
    "gibbon",
    "mitten",
    "trombone",
    "dalmatian",
    "Scotch_terrier",
    "mantis",
    "common_newt",
    "hog",
    "guinea_pig",
    "school_bus",
    "scuba_diver",
    "pomegranate",
    "French_bulldog",
    "scorpion",
    "grand_piano",
    "birdhouse",
    "volcano",
    "whippet",
    "ballplayer",
    "llama",
    "spider_web",
    "sea_lion",
    "ant",
    "tiger",
    "rugby_ball",
    "gazelle",
    "Yorkshire_terrier",
    "jeep",
    "African_chameleon",
    "cocker_spaniel",
    "eel",
    "cheeseburger",
    "German_shepherd",
    "great_white_shark",
    "hammerhead",
    "tabby",
    "koala",
    "cannon",
    "red_fox",
    "space_shuttle",
    "ice_bear",
    "bison",
    "Border_collie",
    "pelican",
    "carousel",
    "hyena",
    "fly",
    "giant_panda",
    "goldfinch",
    "timber_wolf",
    "beaver",
    "electric_guitar",
    "sandal",
    "stingray",
    "baboon",
    "zebra",
    "Pembroke",
    "American_egret",
    "gasmask",
    "meerkat",
    "binoculars",
    "snow_leopard",
    "castle",
    "pirate",
    "centipede",
    "common_iguana",
    "beer_glass",
    "soccer_ball",
    "junco",
    "assault_rifle",
    "Granny_Smith",
    "lipstick",
    "stole",
    "harp",
    "dragonfly",
    "submarine",
    "tennis_ball",
    "vulture",
    "drake",
    "basset",
    "snail",
    "guillotine",
    "missile",
    "broom",
    "leopard",
    "cheetah",
    "accordion",
    "banana",
    "jellyfish",
    "revolver",
    "puffer",
    "axolotl",
    "goldfish",
    "Rottweiler",
    "cockroach",
    "Labrador_retriever",
    "goose",
    "pineapple",
    "West_Highland_white_terrier",
    "skunk",
    "American_lobster",
    "canoe",
    "king_penguin",
    "basketball",
    "wood_rabbit",
    "chow",
    "backpack",
    "anemone_fish",
    "hatchet",
    "fire_engine",
    "chimpanzee",
    "ice_cream",
    "hummingbird",
    "mushroom",
    "hen",
    "beagle",
    "tractor",
    "lab_coat",
    "cucumber",
    "lawn_mower",
    "hammer",
    "bald_eagle",
    "Indian_cobra",
    "wine_bottle",
    "pretzel",
    "bow_tie",
    "barn",
    "hotdog",
    "steam_locomotive",
    "gorilla",
    "acorn",
    "violin",
    "Chihuahua",
    "ambulance",
    "bucket",
    "bee",
    "Weimaraner",
    "pizza",
    "bathtub",
    "cowboy_hat",
    "beacon",
    "badger",
    "flute",
    "tree_frog",
    "toucan",
    "collie",
    "mailbox",
    "Italian_greyhound",
    "boxer",
    "monarch",
    "joystick",
    "hermit_crab",
    "killer_whale",
    "barrow",
    "shield",
    "caldron",
    "schooner",
    "flamingo",
    "tank",
    "black_swan",
    "orangutan",
]

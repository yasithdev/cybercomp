import os
from argparse import ArgumentParser

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block, vit_base_patch16_224
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16
from torchvision.models.vision_transformer import EncoderBlock
from transformers import ConvNextForImageClassification


def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name, patch_size):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # print(state_dict.keys())
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        # remove `encoder.` prefix induced by MAE
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))
    else:
        print("There is no reference weights available for this model => We use random weights.")


def get_ssl_model(model: str):

    print(f"Loading model {model}")

    if model == "dino":
        model_name = "vit_base_patch16_224.dino"

        model = timm.create_model(model_name, pretrained=True)
    elif model == "vit":
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    elif "convnextv2" in model:
        # model_name = "convnextv2_atto.fcmae_ft_in1k"
        model_name = "convnextv2_base.fcmae_ft_in1k"
        model = timm.create_model(model_name, pretrained=True)
    elif "convnextv1" in model:
        model_name = "facebook/convnext-base-224"
        # model = timm.create_model(model_name, pretrained=True)
        model = ConvNextForImageClassification.from_pretrained("facebook/convnext-base-224")

    elif model == "mae":
        model_name = "vit_base_patch16_224.mae"
        model = timm.create_model(model_name, pretrained=True)

    elif model == "mugs":
        model = vit_base_patch16_224()
        path = "weights/mugs_vit_base_backbone_400ep.pth"
        load_pretrained_weights(model, path, "state_dict", "vit_b_16", 16)
    elif model == "resnet50_sl":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    elif model == "resnet50":
        path = "weights/swav_800ep_pretrain.pth.tar"
        model = resnet50()
        state_dict = torch.load(path, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                print('key "{}" could not be found in provided state dict'.format(k))
            elif state_dict[k].shape != v.shape:
                print('key "{}" is of different shape in model and provided state dict'.format(k))
                state_dict[k] = v
        model.load_state_dict(state_dict, strict=False)
    return model


def hook_conv(module, input, output):
    print(f"output shape: {output.shape}")
    output = F.adaptive_avg_pool2d(output, (2, 2))
    return output


def attach_hook(model, target_block, layer):
    count = 1
    attached = False

    for name, module in model.named_modules():
        if isinstance(module, target_block):
            if count == layer:
                module.register_forward_hook(hook_conv)
                attached = True
                break
            count += 1

    if not attached:
        raise ValueError("Layer not found")

    else:
        print(f"Layer {layer} attached")
    return model


def feature_processor(model_name):
    def conv_processor(input):
        output = torch.flatten(F.adaptive_avg_pool2d(input, 2).squeeze(), 1).detach()
        return output

    def transformer_processor(input):
        tokens_except_cls = input[:, 1:, :]
        avg_tokens = torch.mean(tokens_except_cls, dim=1).detach()
        return avg_tokens

    if "resnet" in model_name or "convnext" in model_name:
        return conv_processor
    else:
        return transformer_processor


def get_names_of_layers(model):
    if "resnet" in model:
        target_block = nn.Conv2d
        model = resnet50()
    elif "convnextv2" in model:
        target_block = nn.Conv2d
        model = get_ssl_model("convnextv2")
    elif "convnextv1" in model:
        target_block = nn.Conv2d
        model = get_ssl_model("convnextv1")
    elif "vit" in model:
        target_block = EncoderBlock
        model = vit_b_16()
    else:

        target_block = Block
        model = vit_base_patch16_224()

    names = []
    for name, module in model.named_modules():
        if isinstance(module, target_block):
            if "downsample" in name or "downsampling" in name:
                continue
            name = name.replace(".", "")
            names.append(name)
    return names

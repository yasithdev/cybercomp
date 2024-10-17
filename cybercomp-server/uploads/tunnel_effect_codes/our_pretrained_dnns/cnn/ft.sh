#!/usr/bin/env bash
EXPT_NAME=FineTune_ResNet18_ImageNet100_224
DATA_DIR=/data/datasets/ImageNet1K
NUM_CLASS=50
NUM_EPOCHS=50
CKPT=./resnet18_imagenet100_224_mc/best_resnet18_imagenet100_224_mc_100.pth
GPU=0,1,2,3

CUDA_VISIBLE_DEVICES=${GPU} python -u main_ft.py \
--data ${DATA_DIR} \
--save_dir ${EXPT_NAME} \
--num_classes ${NUM_CLASS} \
--image_size 224 \
--lr 0.01 \
--wd 5e-2 \
--epochs ${NUM_EPOCHS} \
--lr_step_size 20 \
--lr_gamma 0.1 \
-b 2048 \
-p 250 \
--pretrained ${CKPT} \
--ckpt_file ${EXPT_NAME}_${NUM_CLASS}.pth \
--expt_name ${EXPT_NAME} > logs/${EXPT_NAME}.log




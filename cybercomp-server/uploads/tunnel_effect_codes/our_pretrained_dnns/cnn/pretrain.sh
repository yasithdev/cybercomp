#!/usr/bin/env bash
EXPT_NAME=VGG11_IN_100c_100s_no_aug
DATA_DIR=/data/datasets/ImageNet-100
SAVE_DIR=./pretrained_vgg_class_samples
NUM_CLASS=100
NUM_SAMPLES=100
INPUT_SIZE=32 # 32, 64, 128, 224
NUM_EPOCHS=100 # 100(VGG-17) and 70(VGG-11), 200 (CIFAR100, VGG-17)
GPU=0,1,2,3

CUDA_VISIBLE_DEVICES=${GPU} python -u pretrain3.py \
--data ${DATA_DIR} \
--save_dir ${SAVE_DIR} \
--num_classes ${NUM_CLASS} \
--num_samples ${NUM_SAMPLES} \
--image_size ${INPUT_SIZE} \
--lr 0.008 \
--wd 5e-2 \
--seed 42 \
--epochs ${NUM_EPOCHS} \
-b 512 \
-p 250 \
--ckpt_file ${EXPT_NAME}_${NUM_CLASS}.pth \
--expt_name ${EXPT_NAME} > logs/${EXPT_NAME}.log





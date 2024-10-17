#!/usr/bin/env bash
EXPT_NAME=CL_resnet18_imagenet100_32
DATA_DIR=/data/datasets/ImageNet1K
NUM_CLASS=50
NUM_EPOCHS=50
GPU=0,1,2,3
# Specify input resolution 32, 224 in image_size

CUDA_VISIBLE_DEVICES=${GPU} python -u main_cl.py \
--data ${DATA_DIR} \
--save_dir ${EXPT_NAME} \
--num_classes ${NUM_CLASS} \
--image_size 32 \
--lr 0.01 \
--wd 5e-2 \
--epochs ${NUM_EPOCHS} \
--lr_step_size 20 \
--lr_gamma 0.1 \
-b 2048 \
-p 250 \
--ckpt_file ${EXPT_NAME}_${NUM_CLASS}.pth \
--expt_name ${EXPT_NAME} > logs/${EXPT_NAME}.log



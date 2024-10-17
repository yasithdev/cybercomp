#!/usr/bin/env bash
EXPT_NAME=CL_resnet18_imagenet100_32_ft2
DATA_DIR=/data/datasets/ImageNet100
NUM_CLASS=50
NUM_EPOCHS=50
GPU=0,1,2,3
CKPT1=./CL_resnet18_imagenet100_32/task_1_CL_resnet18_imagenet100_32_50c.pth
CKPT2=./CL_resnet18_imagenet100_32/task_2_CL_resnet18_imagenet100_32_50c.pth


CUDA_VISIBLE_DEVICES=${GPU} python -u main_ft2.py \
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
--pretrained_task1 ${CKPT1} \
--pretrained_task2 ${CKPT2} \
--ckpt_file ${EXPT_NAME}_${NUM_CLASS}.pth \
--expt_name ${EXPT_NAME} > logs/${EXPT_NAME}.log



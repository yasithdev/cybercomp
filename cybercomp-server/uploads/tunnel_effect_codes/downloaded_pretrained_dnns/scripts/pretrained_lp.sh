
# MODEL_NAMES=(resnet50)
MODEL_NAMES=(convnextv2 resnet50)
lr=3e-5
for MODEL_NAME in ${MODEL_NAMES[@]}; do


    

    # Imagenet(ID)
    python lp.py --data_dir /data/datasets/ImageNet2012 --dataset_name imagenet --num_classes 1000 --num_classes_pretrained 10   --model_name $MODEL_NAME --lr $lr 

    # Imagenet-r
    python lp.py --data_dir /home/yousuf/dualprompt-pytorch/data/imagenet-r --dataset_name imagenet-r --num_classes 200 --num_classes_pretrained 10   --model_name $MODEL_NAME --lr $lr 



    ## CIFAR100
    python lp.py --data_dir /data/datasets/CIFAR100 --dataset_name CIFAR100 --num_classes 100 --num_classes_pretrained 10   --model_name $MODEL_NAME  --lr $lr 

    ## Pets
    python lp.py --data_dir data/datasets/Pets37 --dataset_name Pets37 --num_classes 37 --num_classes_pretrained 10   --model_name $MODEL_NAME --lr $lr

    ## FGVC Aircraft
    python lp.py --data_dir data/datasets/FGVCAircraft --dataset_name FGVCAircraft --num_classes 102 --num_classes_pretrained 10   --model_name $MODEL_NAME --lr $lr


    ## SLT10
    python lp.py --data_dir data/datasets/SLT10 --dataset_name SLT10 --num_classes 10 --num_classes_pretrained 10   --model_name $MODEL_NAME --lr $lr


    ## Flowers102
    python lp.py --data_dir data/datasets/Flowers102 --dataset_name Flowers102 --num_classes 102 --num_classes_pretrained 10   --model_name $MODEL_NAME --lr $lr

    ## NINCO
    python lp.py --data_dir /home/kyungbok/tunnel_new/src/data/datasets/NINCO/NINCO_OOD_classes --dataset_name NINCO --num_classes 64 --num_classes_pretrained 10   --model_name $MODEL_NAME  --lr $lr 

    # CUB
    python lp.py --data_dir data/datasets/CUB --dataset_name CUB --num_classes 200 --num_classes_pretrained 10   --model_name $MODEL_NAME  --lr $lr 
done 

MODEL_NAMES=(convnextv1 dino mae mugs resnet50_sl vit)
lr=5e-4
for MODEL_NAME in ${MODEL_NAMES[@]}; do


    

    # Imagenet(ID)
    python lp.py --data_dir /data/datasets/ImageNet2012 --dataset_name imagenet --num_classes 1000 --num_classes_pretrained 10   --model_name $MODEL_NAME --lr $lr 

    # Imagenet-r
    python lp.py --data_dir /home/yousuf/dualprompt-pytorch/data/imagenet-r --dataset_name imagenet-r --num_classes 200 --num_classes_pretrained 10   --model_name $MODEL_NAME --lr $lr 



    ## CIFAR100
    python lp.py --data_dir /data/datasets/CIFAR100 --dataset_name CIFAR100 --num_classes 100 --num_classes_pretrained 10   --model_name $MODEL_NAME  --lr $lr 

    ## Pets
    python lp.py --data_dir data/datasets/Pets37 --dataset_name Pets37 --num_classes 37 --num_classes_pretrained 10   --model_name $MODEL_NAME --lr $lr

    ## FGVC Aircraft
    python lp.py --data_dir data/datasets/FGVCAircraft --dataset_name FGVCAircraft --num_classes 102 --num_classes_pretrained 10   --model_name $MODEL_NAME --lr $lr


    ## SLT10
    python lp.py --data_dir data/datasets/SLT10 --dataset_name SLT10 --num_classes 10 --num_classes_pretrained 10   --model_name $MODEL_NAME --lr $lr


    ## Flowers102
    python lp.py --data_dir data/datasets/Flowers102 --dataset_name Flowers102 --num_classes 102 --num_classes_pretrained 10   --model_name $MODEL_NAME --lr $lr

    ## NINCO
    python lp.py --data_dir /home/kyungbok/tunnel_new/src/data/datasets/NINCO/NINCO_OOD_classes --dataset_name NINCO --num_classes 64 --num_classes_pretrained 10   --model_name $MODEL_NAME  --lr $lr 

    # CUB
    python lp.py --data_dir data/datasets/CUB --dataset_name CUB --num_classes 200 --num_classes_pretrained 10   --model_name $MODEL_NAME  --lr $lr 
done 

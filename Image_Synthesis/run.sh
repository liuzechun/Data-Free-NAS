CUDA_VISIBLE_DEVICES=1 python train.py --batch_size=50 --exp_name="resnet50_recursive" --r_feature=0.01 --arch_name="resnet50" --verifier --lr 0.25 --store_best_images

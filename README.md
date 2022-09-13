# Data-Free-NAS
This is the pytorch implementation of our paper ["Data-Free Neural Architecture Search via Recursive Label Calibration"](https://arxiv.org/abs/2112.02086), published in ECCV 2022.

<div align=center>
<img width=60% src="https://github.com/liuzechun0216/images/blob/master/data-free_NAS_github.jpg"/>
</div>

This paper aims to explore the feasibility of neural architecture search (NAS) without original data, given only a pre-trained model. Our results demonstrate that the architectures discovered by our data-free NAS achieve comparable accuracy as architectures searched from the original natural data. This derives the conclusion that NAS can be done effectively and data-freely.

## Citation

If you find our code useful for your research, please consider citing:

    @inproceedings{liu2022data,
      title={Data-Free Neural Architecture Search via Recursive Label Calibration},
      author={Liu, Zechun and Shen, Zhiqiang and Long, Yun and Xing, Eric and Cheng, Kwang-Ting and Leichner, Chas},
      booktitle={European Conference on Computer Vision (ECCV)},
      year={2022}
    }

## Run

### 1. Requirements:
* python 3.6, pytorch 1.7.1, torchvision 0.8.2

### 2. Steps to run:

(1) Step 1: image synthesis
* Put the pretrained [ResNet-50](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EcKz2vfk51NNvU8hxohlNUgB5R5r2Tn5qECtqr3JegtcKA?e=b8ML2o) into the folder `./Image_Synthesis/models` 
* Change directory to `./Image_Synthesis/` 
* run `bash run.sh`

(2) Step 2:  neural architecture search

Step 2.0: split the synthesized data into the training set for supernet training and validation set for evolutionary search
* Change directory to `./NAS/`
* Run `python split.py`

Step 2.1: supernet training
* Change directory to `./NAS/Supernet/`
* Run `bash run.sh`

Step 2.2: evolutionary search
* Change directory to `./NAS/Search/` 
* Run `bash run_search.sh` 

Step 2.3: evaluation
* Change directory to `./NAS/Evaluation/` 
* Run `bash run_eval.sh` and it will automatically generate a folder containing the searched architecture and the corresponding training code in `./data`.
* Change directory to `./NAS/Evaluation/data/${architecture}` 
* Run `python train.py`

## Models and synthesized data

### 1. Pretrained ResNet-50 model: [ResNet-50](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EcKz2vfk51NNvU8hxohlNUgB5R5r2Tn5qECtqr3JegtcKA?e=b8ML2o)

### 2. Synthesized images: [Data](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EfattltEMwtLgPy7AGicGYUBtdHZpJSWr8L60ON3dfjBIg?e=jnyEh5)

### 3. Searched model and final results:

| Methods | Top1-Err (%) | FLOPs (M) | Data for NAS |
| --- | --- | --- | --- | 
| Single Path One-Shot (SPOS) | 25.7 | 319 | ImageNet |
| [Data-Free SPOS](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/ET5ro_8mABVGohjFWBrJjBMB2iZyp0gE1T87ALwEQYhXiw?e=ABr5eD) | 25.8 | 316 | Synthesized data | 

## Contact

Zechun Liu, HKUST (zliubq at connect.ust.hk)

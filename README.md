# Barebone implementation of DDPM

Simple Implementation of [DDPM](https://arxiv.org/abs/2006.11239). 


## Usage

### Setup Environment

DDPM is very computationally expensive. This repo requires CUDA. 
If GPU does not have enough memory, try to reduce batch size.

```
conda env create -f environment.yml
conda activate ddpm_torch
python main.py -h
```

### Train model

Current code supports two datasets: CIFAR10 and FashionMNIST.

```
python main.py train -h
python main.py train --dataset=cifar10
python main.py train --dataset=fashion
```

Additional flags such as `batch_size`, `epochs`, `timesteps` and checkpoint intervals `ckpt_interval`.

The model will start training and saving model weights to `./checkpoints/`.


### To Sample

```
python main.py infer -h
python main.py infer <epoch> --sample_n=16 --dataset=cifar10
```

Using the last checkpoint (`cifar10_epc_999.pt`) to sample some images.
Images will be saved in `./images/`.

One can combine all 16 sample trajectories using [imagemagic](imagemagick.org).
```
montage -density 300 -tile 16x0 -geometry +1+1 -border 2 images/*.png out.png
```

## More complete implementation of DDPM

* author's tensorflow https://github.com/hojonathanho/diffusion
* pytorch version: https://github.com/lucidrains/denoising-diffusion-pytorch

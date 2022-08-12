# Barebone implementation of DDPM

Simple Implementation of DDPM. 
No moving mean average. 

```
python main.py train cifar10
```
This will train cifar10 for 1000 epochs with batch size 256.
At the end of every epoch, a checkpoint file will be saved.


### To Sample
```
python main.py infer 999
```

Using the last checkpoint (`e_999.pt`) to sample some images.

To combine all images 
```
montage -density 300 -tile 16x0 -geometry +1+1 -border 2 images/*.png out.png
```

## More complete implementation of DDPM

repo 1:
repo 2:

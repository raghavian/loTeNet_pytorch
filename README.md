# README #

This is official Pytorch implementation of LoTeNet model in 
"[Tensor Networks for Medical Image Classification](https://openreview.net/forum?id=jjk6bxk07G)", Raghavendra Selvan & Erik Dam, MIDL 2020

![lotenet](utils/model.png)
### What is this repository for? ###

* Run and reproduce results in the paper on LIDC dataset
* v1.0


### How do I get set up? ###

* Basic Pytorch dependency
* Tested on Pytorch 1.3, Python 3.6 
* Unzip the data and point the path to --data_path
* How to run tests: python train.py --data_path data_location

### Usage guidelines ###

* Kindly cite our publication if you use any part of the code

```
@inproceedings{
raghav2020tensor,
title={Tensor Networks for Medical Image Classification},
author={Raghavendra Selvan, Erik B Dam},
booktitle={International Conference on Medical Imaging with Deep Learning -- Full Paper Track},
year={2020},
month={July}
url={https://openreview.net/forum?id=jjk6bxk07G}}
```

### Who do I talk to? ###

* raghav@di.ku.dk

### Thanks to the following repositories we base our project on:
* [Torch MPS](https://github.com/jemisjoky/TorchMPS/) for the amazing MPS in Pytorch implementations
* [Prob.U-Net](https://github.com/stefanknegt/Probabilistic-Unet-Pytorch) for preprocessing LIDC data
* [Dense Net](https://github.com/bamos/densenet.pytorch/) implementation

# pytorch implementation of SemiAdv
This repository contains code of SemiAdv ([SemiAdv: Query-Efficient Black-Box Adversarial Attack with Unlabeled Data]()) implemented in Pytorch.

# Requirements
+ Python 3.9.2
+ Pytorch 1.9
+ Torchvision 0.1.8

# Instructions

## Quick Start
We prepare an easy demo for quick start.
```
python train_black_model.py
python train_substitute_model.py
python attack.py
```
The train_black_model.py contains the code of training a black-box model.

The train_substitute_model.py contains the code of training a substitute model of the black-box model.

The attack.py contains the code of implementing black-box attack by using the sustitute model against the black-box model.

The default setting is as follows:
  + black model: MobileNet,
  + sustitute model: WideResNet-28,
  + default attack method: PGD with our method,
  + dataset: CIFAR-10,
  + query number (labeled data): 1600.
  
More defulat setting or info refers to source code.

## Training black-box model

## Training substitute model
More information will quickly arrive.

## Attack
More information will quickly arrive.

## Others
customizing your model or dataset:

If you have any questions, please contact us or leave a message here.

# Citing this paper


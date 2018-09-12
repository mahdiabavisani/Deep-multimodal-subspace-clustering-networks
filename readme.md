# Deep multimodal subspace clustering networks

![overview](./fig2.pdf)


## Overview
This repository contains the implementation of the paper "Deep multimodal subspace clustering networks" by Mahdi Abavisani and Vishal M. Patel. The paper was posted on arXiv in May 2018.

"Deep multimodal subspace clustering networks" (DMSC)  investigated various fusion methods for the task of multimodal subspace clustering, and suggested a new fusion technique called "affinity fusion" as the idea of integrating complementary information from two modalities with respect to the similarities between datapoints across different modalities. 

![overview](./fig1.pdf)



## Setup:
#### Dependencies:
Tensorflow, numpy, sklearn, munkres, scipy.
#### Data preprocessing:
Resize the input images of all the modalities to 32 × 32, and rescale them to have pixel values between 0 and 255.   This is for keeping the hyperparameter selections suggested in [Deep subspace clustering networks](https://github.com/panji1990/Deep-subspace-clustering-networks) valid. 

Save the data in a `.mat` file that includes verctorized modalities as separate matrices with the names `modality_0`,`modality_1`, ... ; labels in a vector with the name `Labels`; and number of modalities in the variable `num_modalities`.

A sample preprocessed dataset is available in: `Data/EYB_fc.mat` 


### Running the code

#### Affinity fusion :
Run `affinity_fusion.py` to do mutlimodal subspace clustering.  For demo a pretrained model trained on `EYB_fc` is avilable in `models/EYBfc_af.ckpt`

Run the demo as: 
```
python affinity_fusion.py --mat EYB_fc --model EYBfc_af
```
#### Pretraining:
Run `pretrain_affinity_fusion.py` to pretrain your networks. 

For example:
```
python pretrain_affinity_fusion.py --mat EYB_fc --model mymodel --epoch 100000
```



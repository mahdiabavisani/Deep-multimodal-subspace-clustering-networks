# Deep multimodal subspace clustering networks

![fig1](https://user-images.githubusercontent.com/18729506/45459073-ab626e00-b6c4-11e8-9271-227e6a0b1f22.jpg)



## Overview
This repository contains the implementation of the paper "Deep multimodal subspace clustering networks" by Mahdi Abavisani and Vishal M. Patel. The paper was posted on arXiv in May 2018.

"Deep multimodal subspace clustering networks" (DMSC)  investigated various fusion methods for the task of multimodal subspace clustering, and suggested a new fusion technique called "affinity fusion" as the idea of integrating complementary information from two modalities with respect to the similarities between datapoints across different modalities. 

![fig1](https://user-images.githubusercontent.com/18729506/45457918-2f195c00-b6bf-11e8-908b-01817a5e3387.jpg)


## Citation

Please use the following to refer to this work in publications:

<pre><code>
@ARTICLE{8488484, 
author={M. {Abavisani} and V. M. {Patel}}, 
journal={IEEE Journal of Selected Topics in Signal Processing}, 
title={Deep Multimodal Subspace Clustering Networks}, 
year={2018}, 
volume={12}, 
number={6}, 
pages={1601-1614}, 
doi={10.1109/JSTSP.2018.2875385}, 
ISSN={1932-4553}, 
month={Dec},}
</code></pre>


## Setup:
#### Dependencies:
Tensorflow, numpy, sklearn, munkres, scipy.
#### Data preprocessing:
Resize the input images of all the modalities to 32 × 32, and rescale them to have pixel values between 0 and 255.   This is for keeping the hyperparameter selections suggested in [Deep subspace clustering networks](https://github.com/panji1990/Deep-subspace-clustering-networks) valid. 

Save the data in a `.mat` file that includes verctorized modalities as separate matrices with the names `modality_0`,`modality_1`, ... ; labels in a vector with the name `Labels`; and number of modalities in the variable `num_modalities`.

A sample preprocessed dataset is available in: `Data/EYB_fc.mat` 


## Running the code

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

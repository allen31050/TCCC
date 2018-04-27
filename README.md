# TCCC
Toxic Comment Classification Challenge

## 1. Blending of Gru Networks
```grus_avg_blending```  
average blending of ```gru_conv```, ```multi_gru_conv```,```capsule_gru_conv```
- ```gru_conv```  
gru with convolution
- ```multi_gru_conv``` [1]  
multi-column deep neural networks (with each column remains gru with convolution) 
- ```capsule_gru_conv``` [2]  
capsule network (with major network remains gru with convolution)

## 2. Out-of-fold (oof) Blending of other State-of-the-art Models
```oof_blending```  
models with oof run and stored on Kaggle kernels, here are only scripts to merge results

## 3. Final Blending
```hierarchy_blending```  
blend results of (1) Blending of Gru Networks and (2) Out-of-fold (oof) Blending of other State-of-the-art Models    
(Note that most data and results are kept offline)


## Source
[1] Ciregan, D., Meier, U., & Schmidhuber, J. (2012, June). Multi-column deep neural networks for image classification. In Computer vision and pattern recognition (CVPR), 2012 IEEE conference on (pp. 3642-3649). IEEE.  
[2] Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic routing between capsules. In Advances in Neural Information Processing Systems (pp. 3859-3869).  
[3] other code sources and modifications inspired by posts in  
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/discussion  
https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/kernels/

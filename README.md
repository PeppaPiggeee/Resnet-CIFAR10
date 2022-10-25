<<<<<<< HEAD
# Deep Residual Learning for Image Recognition

This repository is the first assignment of my AI security course, which is a simple PyTorch implementation of 34 - layer ResNet on CIFAR10.



## Requirements

```
conda create -n resnet python=3.7
conda activate resnet
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

Please go to [the official website](https://pytorch.org/) and download torch which matches your CUDA version.



## Train

```
CUDA_VISIBLE_DEVICES=0 python run.py --resume_last --resume_best
```



## Test

```
CUDA_VISIBLE_DEVICES=0 python run.py --resume_last --resume_best --test
```



## Citation

> ```
> @inproceedings{he2016deep,
>   title={Deep residual learning for image recognition},
>   author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
>   booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
>   pages={770--778},
>   year={2016}
> }
> ```
=======
# Deep Residual Learning for Image Recognition

This repository is the first assignment of my AI security course, which is a simple PyTorch implementation of 34 - layer ResNet on CIFAR10.



## Requirements

```
conda create -n resnet python=3.7
conda activate resnet
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

Please go to [the official website](https://pytorch.org/) and download torch which matches your CUDA version.



## Train and Test

```
CUDA_VISIBLE_DEVICES=0 python run.py 
```

Run the above command. The loss and accuracy of training and testing can be found in result.png.



## Citation

> ```
> @inproceedings{he2016deep,
> title={Deep residual learning for image recognition},
> author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
> booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
> pages={770--778},
> year={2016}
> }
> ```
>>>>>>> 76025cf8476bad10ea18d548b2670aa5d86e0f1e

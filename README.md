# Neural Networks Quantization
This repository contains implementations of two quantization algorithms for arrays of floating point numbers, especially
neural networks parameters. Moreover, this repository provides simple script performing the quantization for deep
learning models defined by the [TensorFlow-Slim](https://github.com/tensorflow/models/tree/master/research/slim).

## Quantization algorithms
The **bin quantization** algorithm assigns the bin number (integer) ranging from *0* to *(1 << bit width) - 1* (for
example, for eight bits, the least bin number is *0*, while the highest bin number is 255) to floating points acting as
an input of this quantization method. The bins are equally distributed (histogram-like) and are bounded by two floating
points. All the bins are left-closed and right opened, except to the last one (which is also right-closed). During the 
dequantization process, all the numbers within given bin are replaced by the same number representing the *center* of
this bin.  
  
The **fixed-point quantization** algorithm represents each of floating points acting as an input of this quantization
method as a fixed point decimal using given number of bits. This method introduces the loss of significance imposed by 
reducing the bit-width.

## Example
This repository contains an example script performing the quantization of two TensorFlow-Slim deep learning models:
[ResNet V1 50](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py) and 
[ResNet V1 101](https://github.com/tensorflow/models/blob/master/research/slim/nets/resnet_v1.py). Follow the next steps
to run the whole example from scratch.  
  
Please note that this example is intended to run the inference on the 
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

### Installation
It is assumed that you have the following tools installed and configured:
- Python 3.5.x (or later);
- TensorFlow 1.0 (or later) library conforming to your Python installation.

In order to run the sample script:
1. Clone this repository.
2. Clone the [TensorFlow-Slim model library](https://github.com/tensorflow/models/tree/master/research/slim) containing
models of deep learning neural networks, as well as some utility scripts used below. Place this repository in the main
directory of this project.

### Downloading the dataset
In order to download the CIFAR-10 dataset in an appropriate format, please execute the following command from the main
directory of this project:
```
python3 models/research/slim/download_and_convert_data.py --dataset_name=cifar10 --dataset_dir=cifar10/
```
The dataset should be placed under the `cifar10/` directory accessed directly from the main directory of this project.

### Downloading the pretrained models
Download the latest versions of 
[Pre-trained Models](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models) of deep learning
neural networks. In this example, two checkpoints are downloaded:
- Resnet V1 50;
- Resnet V1 101.

It is assumed that these checkpoint files are placed under the `checkpoints/` directory and are named, respectively, 
`resnet_v1_50.ckpt` and `resnet_v1_101.ckpt`.  
  
Please note that the appropriate checkpoint file may be downloaded via the command line by following the steps described
[here](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models).

### Fine-tuning
The TensorFlow-Slim models are trained on the ImageNet dataset with 1000 class labels, while the CIFAR-10 dataset has
only 10 classes. Therefore, it is a common trick to drop the final layer of any pretrained model (usually called 
`Logits`), replace it with the new one - having appropriate number of neurons - and train it (using the new dataset)
with all weights frozen, except to these connected to the newly-created layer. This process of "training" is called fine-tuning.  
  
In order to fine-tune the ResNet V1 50 network from the TensorFlow-Slim image classification model library with the
CIFAR-10 dataset, simply type (in the main directory of this project):
```
python3 models/research/slim/train_image_classifier.py
    --clone_on_cpu=True
    --max_number_of_steps=1
    --train_dir=finetuning/resnet_v1_50/
    --dataset_dir=cifar10/
    --dataset_name=cifar10
    --dataset_split_name=train
    --model_name=resnet_v1_50
    --checkpoint_path [ABSOLUTE PATH TO THE MAIN DIRECTORY]/checkpoints/resnet_v1_50.ckpt
    --checkpoint_exclude_scopes=resnet_v1_50/logits
    --trainable_scopes=resnet_v1_50/logits
```
If you want to fine-tune the network on your GPU, please remove the `--clone_on_cpu=True` flag. The 
`--max_number_of_steps` flag indicates the number of steps to be performed by the training algorithm (here, 1 is set for
simplicity). The output of the fine-tuning process will be saved to the `finetuning/resnet_v1_50/` directory indicated by
the `--train_dir` flag. The `--dataset_dir` flag points the localization of downloaded dataset, while the `--dataset_name`
flag indicates the name of this dataset. Fine-tuning is a training operation, therefore the `train` part of the CIFAR-10
dataset is used, as passed by the `--dataset_split_name` flag. As said before, the Resnet V1 50 network is under training,
therefore the appropriate value for the `--model_name` flag is set. The `--checkpoint_path` flag points the absolute
path to the checkpoint file downloaded before. As described above, the final `Logits` layer should be dropped while
restoring the checkpoint (therefore, the `--checkpoint_exclude_scopes` flag is used), and the new final layer should be
created and trained (thus, the `--trainable_scopes` flag points the name of the `Logits` layer).  
  
A modification of the command above for the ResNet V1 101 network architecture is straightforward.

### Quantization
Quantization algorithms convert the `float32` representation of floating-point numbers (here, the trainable variables,
i.e. weights and batch norms) to another representation of given `bit width`. This representation is either integer
(in case of the bin quantization algorithm) or decimal (in case of the fixed-point quantization algorithm).  
  
In order to run the quantization of trainable variables belonging to previously fine-tuned models, type the following
command with appropriate arguments (from the main directory of this project):
```
python3 quantizer.py [NETWORK MODEL] [CHECKPOINT FILE PATH] [NUMBER OF CLASSES] [QUANTIZATION ALGORITHM] [BIT WIDTH]
```
The `[NETWORK MODEL]` argument codes the ResNet V1 architecture to be used - `50` corresponds to the ResNet V1 50, while
`101` corresponds to the ResNet V1 101 network. The `[CHECKPOINT FILE PATH]` argument indicates the relative path to the 
file holding the checkpoint file produced as an output of the fine-tuning process described above. The `[NUMBER OF
CLASSES]` argument points the number of neurons in the `Logits` layer (please note that this number usually corresponds 
to the number of different class labels in the dataset, but not always - for example, for the Inception network and the 
ImageNet dataset, the number of neurons should be 1001, while the number of classes is 1000). The `[QUANTIZATION
ALGORITHM]` argument codes the quantization method to be used - `1` corresponds to the bin quantization algorithm, while
`2` corresponds to the fixed-point quantization algorithm. The `[BIT WIDTH]` algorithms indicates the number of bits 
used to represent the quantized values.  
  
In order to apply the 8-bits bin quantization method to the ResNet V1 50 model fine-tuned on the CIFAR-10 dataset, type
the following command:
```
python3 quantizer.py 50 "finetuning/resnet_v1_50/model.ckpt-1" 10 1 8
```

Please note that the quantization script provides you with some relative errors summary. The error is computed as the
quotient of the Euclidean norm of the difference between the input vector and the quantized vector, to the Euclidean
norm of the original (input) vector. The script in question summarizes the minimal, maximal and average error for all
trainable variables defined by given network architecture.  
  
The quantization script saves a series of checkpoint files containing new values for model variables in a directory 
named `[QUANTIZATION ALGORITHM NAME]_[BIT WIDTH]bits`, placed under the directory containing the checkpoint file to be 
quantized (and pointed by the `[CHECKPOINT FILE PATH]` argument). For example, when performing a bin quantization 
algorithm to 8 bits, applied to the ResNet V1 50 model checkpointed to the `finetuning/resnet_v1_50/model.ckpt-1` file, 
an output of the quantization algorithm will be placed under the `finetuning/resnet_v1_50/bin_8bits/` directory.

### Evaluation
The evaluation script provided by the TensorFlow-Slim image classification model library computes values of two common
machine learning metrics: TOP-1 Accuracy and TOP-5 Recall.  
  
In order to run the evaluation script for the 8-bits bin-quantized ResNet V1 50 model fine-tuned on the CIFAR-10 dataset,
type the following command (from the main directory of this project):
```
python3 models/research/slim/eval_image_classifier.py
    --checkpoint_path=finetuning/resnet_v1_50/bin_8bits/
    --dataset_dir=cifar10/
    --dataset_name=cifar10
    --dataset_split_name=test
    --model_name=resnet_v1_50
```
The `--checkpoint_path` flag points the directory containing a checkpoint of the fine-tuned (and quantized, or not)
network model (here, the `finetuning/resnet_v1_50/bin_8bits` directory contains the checkpoint file produced by the bin
quantization algorithm to 8 bits, applied to the ResNet V1 50 network fine-tuned on the CIFAR-10 dataset). Description
of the other flags has been presented above. It is necessary to mention that the evaluation should be performed on the 
`test` part of the dataset, therefore the `--dataset_splot_name` flag value is set accordingly.  
  
Values of performance evaluation metrics are logged on the standard output.  
  
A modification of the command above is straightforward for the ResNet V1 101 network architecture and other quantization
methods (i.e. the fixed-point quantization algorithm or any algorithm with other bit width).

### Troubleshooting
Please refer to the instructions linked in the *References* section below.  


## References:
[1] [TensorFlow-Slim library](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)  
[2] [TensorFlow-Slim image classification model library](https://github.com/tensorflow/models/tree/master/research/slim)  
[3] [TensorFlow-Slim Walkthrough](https://github.com/tensorflow/models/blob/master/research/slim/slim_walkthrough.ipynb)  
[4] [TensorFlow docs](https://www.tensorflow.org/api_docs/)

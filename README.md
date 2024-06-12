
## What is this?


This is an example of the implementation of a new approach to quantization of neural networks using the example of the ResNet50 network and the ImageNET1000 dataset.  More links to work will come later

## Original ResNet50 v1 architecture

Manual Resnet50 v1.5 implemented in [file](./resnet50.py). Quantized versions of ReLU and Sigmoid activation finction is plased [here](./qrelu.py)

![ResNet architechture](./images/resnet_arch)


## Install and test

Only install:

```
poetry install --without dev
```


Install for develop (test, linter, formater, pre-commite and other will be installed):

```
poetry install
```

Run the tests:

```
poetry run python -m pytest -vv ./tests/model/model_tests.py --cov
```

## Load data scripts

Download dataset (if need):

`poetry run python getdataset.py -d ImageNet --only-val`

Run the extract script:

`bash ./scripts/extract_ILSVRC_only_val.sh`


## Train scripts


Run the train script:

`poetry run python train.py -m manual`

## Usage:

```
usage: train.py [-h] [-m {torch,manual,quantized,quantizedBest}] [-qa {qRelu,qSigmoid}] [-qb QUANT_BIT]                
                [-qpr QUANT_PERCENT_REDUSE] [-qr QUANT_RANGE] [--opt {sgd,sgd_nesterov,rmsprop,adamw}] [--lr LR]       
                [--label-smoothing LABEL_SMOOTHING] [--epochs N] [-b BATCH_SIZE] [-j N] [--device DEVICE]              
                [--num_classes NUM_CLASSES] [--val_resize_size VAL_RESIZE_SIZE] [--data-path DATA_PATH]                
                [--statistic] [--checkpoints-dir CHECKPOINTS_DIR] [--cache-dataset] [--print-freq PRINT_FREQ]          
                [--log-vandb]

Work to test new quantization strategy

options:                                                   
  -h, --help            show this help message and exit
  -m {torch,manual,quantized,quantizedBest}, --model {torch,manual,quantized,quantizedBest}                            
                        What the model use: standart torch, manual, quantized                                          
  -qa {qRelu,qSigmoid}, --quant-activation {qRelu,qSigmoid}                                                            
                        Type quantized activation: qRelu or qSigmoid                                                   
  -qb QUANT_BIT, --quant-bit QUANT_BIT
                        Size of quantization
  -qpr QUANT_PERCENT_REDUSE, --quant-percent-reduse QUANT_PERCENT_REDUSE                                               
                        Percent of quantization reduse
  -qr QUANT_RANGE, --quant-range QUANT_RANGE
                        Range of quantization
  --opt {sgd,sgd_nesterov,rmsprop,adamw}
                        optimizer. Only four: SGD, SGD with Nesterov, RMSprop or AdamW                                 
  --lr LR               learning rate
  --label-smoothing LABEL_SMOOTHING
                        label smoothing (default: 0.0)
  --epochs N            number of total epochs to run
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        total batch size
  -j N, --workers N     number of data loading workers (default: 4)                                                    
  --device DEVICE       device (Use cuda or cpu Default: cuda)                                                         
  --num_classes NUM_CLASSES                                
                        number of classes
  --val_resize_size VAL_RESIZE_SIZE
                        Size image for validation. Equal for test                                                      
  --data-path DATA_PATH                                    
                        dataset path or name for existing
  --statistic           return list with statistic and train paramrters                                                
  --checkpoints-dir CHECKPOINTS_DIR
                        path to save checkpoints
  --cache-dataset       Cache the datasets for quicker initialization. It also serializes the transforms               
  --print-freq PRINT_FREQ                                  
                        print frequency
  --log-vandb           Begin loggin with wandb. The <ID> must be locate in root dir in file 'wan.db'      


```


## Notebooks with examples

For run in Kaggle use code from  [jupiter-notebook](./notebooks/train_example_on_Kaggle.ipynb)




## Links

https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

https://github.com/pytorch/vision/blob/main/references/classification

https://github.com/robmarkcole/resent18-from-scratch/blob/main/resnet18.py

https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch/files

https://github.com/lyzustc/Numpy-Implementation-of-ResNet/blob/master/network.py

https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh

https://github.com/GokuMohandas/Made-With-ML

https://jmlr.org/papers/volume25/18-566/18-566.pdf

https://kaifishr.github.io/2020/04/29/neural-networks-and-step-activation-functions.html

https://arxiv.org/abs/1312.4400

https://github.com/PiotrDabkowski/torchpwl

https://efficientml.ai/

https://jmlr.org/papers/v25/18-566.html

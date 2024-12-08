# LPRNet_Optimization
This project is an extension to the existing project [LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch). In this project our main focus is to optimize an existing DNN model, LPRNet, to efficiently recognize vehicle plates. Our optimizations try to reduce the model size and improve its running speed while keeping a high accuracy. We are majorly using the following optimization techniques:
1. **Model Compression**  
  - Unstrucutred Pruning
  - Post Training Static Quantization (PTSQ)
2. **Model Compilation Optimization**
  - 23 Compiler level optimization using Apache TVM's autoscheduler
    

## Prerequisites
1. Clone this repo:
```
 git clone https://github.com/sgandhi31/LPRNet_Optimization.git
```
3. Open the LPRNet_Optimization directory.
 ```
 cd LPRNet_Optimization
 ```
3. (Optional) setup a virtual environment to install necessary packages using the following command:
``` commandline
python3 -m venv venv
source .venv/bin/activate
```
4. Install the packages listed in Requirements.txt file
```shell
pip install -r requirements.txt
```

## Project Structure

- data: This directory contains test images and data_load helper scripts.
- mlc: This directory contains script to apply TVM's Machine Learning Compilation Optimizations
- model: This directory contains scripts to get original model and its pruned as well quantized versions.
- weigths: Saved pretrained Pytorch models.
- README.md: you are refering this.
- requirements.txt: all the necessary installation guide.
- test_LPRNet.py: This script tests the original version of LPRNet model.
- test_LPRNetOpt_mlc.py: This script tests the mlc optimized model.
- test_LPRNet_quantize.py: This script tests the quantized version of the model.
- train_LPRNet.py: Script for training the LPRNet model.
- tuning_log.json: File to compile MLC optimizations. 


## Getting Started

This repository mainly focuses on the inference time optimizations of the LPRNet model. We are focusing mainly on 2 model compression techniques and using Apache TVM for Machine Learning Compilation optimization. 

1. Test original model's performance
```
!python3 test_LPRNet.py --test_img_dirs 'data/test' --pretrained_model 'weights/Final_LPRNet_model.pth'
```
2. Test quantized model's performance
```
!python3 test_LPRNet_quantize.py --test_img_dirs 'data/test' --quantized_model 'weights/Final_LPRNet_model.pth'
```
3. Test MLC optimized model's performance
```
!python3 test_LPRNetOpt_mlc.py --test_img_dirs 'data/test' --optimized_model 'weights/Final_LPRNet_model.pth' --log_file '/content/LPRNet_Pytorch/tuning_log.json'
```
4. Test pruned model's performance
```
!python3 test_LPRNet.py --test_img_dirs 'data/test' --pretrained_model '/content/LPRNet_Optimization/weights/unstructured_pruned_LPRNet_model_0.5.pth'
```
5. Test pruned+quantized model's performance
```
!python3 test_LPRNet_quantize.py --test_img_dirs 'data/test' --quantized_model '/content/LPRNet_Optimization/weights/unstructured_pruned_LPRNet_model_0.5.pth'
```
6. Test pruned + MLC optimized model's performance
```
!python test_LPRNetOpt_mlc.py --test_img_dirs 'data/test' --optimized_model '/content/LPRNet_Optimization/weights/unstructured_pruned_LPRNet_model_0.5.pth' --log_file '/content/LPRNet_Pytorch/tuning_log.json'
```

For more detailed information of different components of this script, please refer to docs. 

## Results
Below are the performance comparison of different variants of the model considering 3 metrics: Test Accuracy, Size of the model, and Inference time. All the tests are conducted on the test dataset provided in this repository and on 'x86' backend.

| **Model Variant**        | **Size (MB)** | **Test Accuracy (%)** | **Inference Time (ms)** |
|---------------------------|---------------|------------------------|--------------------------|
| Original (baseline)       | 1.91          | 90.3                  | 212                      |
| Quantized                 | 0.5           | 78.0                  | 27                       |
| Pruned                    | 1.91          | 82.5                  | 152                      |
| MLC                       | 1.91          | 89.0                  | 82                       |
| Pruned + Quantized        | 0.5           | 62.5                  | 23                       |
| Pruned + MLC              | 1.91          | 80.0                  | 68                       |


## Contributors
You may contact the author for any support. We will try to actively monitor the issues raised. Thanks!

[Shyamal Gandhi](sgandhi6@ncsu.edu)


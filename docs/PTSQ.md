# LPRNet Optimization with Post Training Static Quantization

This directory contains scripts for applying post-training quantization to the **LPRNet** model and evaluating its performance. Quantization reduces model size and improves inference speed, making it efficient for deployment on edge devices.

---

## Scripts Overview

### 1. **`LPRNet_quantize.py`**

This script defines the quantized version of the LPRNet model, incorporating PyTorch's quantization utilities. The script includes the following components:

- **`small_basic_block` Class**:  
  A building block for the LPRNet backbone, consisting of convolutional layers with ReLU activations.

- **`LPRNet` Class**:  
  Defines the LPRNet architecture, including support for quantization using `QuantStub` and `DeQuantStub`.

- **`build_lprnet_quantize` Function**:  
  A helper function to initialize and return the LPRNet model with quantization support, ready for evaluation.

---

### 2. **`test_LPRNet_quantize.py`**

This script tests the quantized LPRNet model and evaluates its accuracy using a test dataset. It includes the following functionality:

- **Argument Parsing (`get_parser`)**:  
  Provides configurable options for image size, batch size, quantized model path, and more.

- **Quantization Workflow**:  
  - **`QuantizedLPRNet` Class**: Wraps the LPRNet model with quantization stubs.
  - **`apply_quantization` Function**: Prepares and calibrates the model for post-training static quantization.
  - **`calibrate_quantized_model` Function**: Loads a pretrained model, calibrates it using a dataset, and returns the quantized version.

- **Testing and Evaluation**:  
  - **`test` Function**: Loads the quantized model and evaluates its accuracy on the test dataset.
  - **`Greedy_Decode_Eval` Function**: Decodes predictions, compares them with ground truth, and calculates test accuracy.
  - **`show` Function**: Displays images with predicted and target labels using matplotlib.

---
Output:
```
1. size : 0.5 MB
2. Accuracy: 78%
3. Inference Time: 0.027 seconds
```

## How to Use

### Quantize the LPRNet Model
1. Prepare a pretrained LPRNet model and a representative calibration dataset.
2. Use `test_LPRNet_quantize.py` to test the quantized model:
   ```bash
    !python3 test_LPRNet_quantize.py --test_img_dirs 'data/test' --quantized_model 'weights/Final_LPRNet_model.pth'
   ```

---

## Notes

- Ensure the test dataset is prepared and follows the expected format (`CHARS`, `CHARS_DICT`).
- Quantization uses PyTorch's `fbgemm` backend for efficient computation.
- Modify image size and batch size parameters in the argument parser for custom datasets.

---

This README serves as a reference for understanding the functionality of the scripts in this directory. For detailed usage, integrate these scripts into the larger project pipeline.

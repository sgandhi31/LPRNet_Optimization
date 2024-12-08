# Unstructured Pruning for LPRNet

This script performs **unstructured pruning** on the LPRNet model to reduce its size and improve efficiency by pruning weights in convolutional and linear layers based on L1-norm. The pruned model can be saved for further use.

---

## Script Overview

### **`LPRNet_pruning.py`**

This script includes the following functionality:

1. **Loading the Pretrained Model:**  
   - The `load_pretrained_model` function loads the LPRNet model from a checkpoint file. It handles different checkpoint structures and ensures the model is set to evaluation mode.

2. **Unstructured Pruning:**  
   - The `unstructured_pruning` function prunes weights in convolutional (`Conv2d`) and linear (`Linear`) layers of the model using PyTorch's L1-norm pruning method.
   - The pruning ratio can be specified as an input argument.

3. **Model Size Analysis:**  
   - The `get_model_size` function calculates the number of non-zero parameters in a model, allowing for comparison between the original and pruned models.

4. **Pruning Workflow:**  
   - The `apply_pruning` function orchestrates the entire pruning process:
     - Loads the pretrained model.
     - Applies unstructured pruning with the specified pruning ratio.
     - Optionally displays the parameter reduction percentage.
     - Saves the pruned model to a specified file.

---

## How to Use

### 1. Prerequisites

Ensure the pretrained LPRNet model is available at the path specified in the `--pretrained_model` argument.


### 1. Running the Script

Use the following command to prune the model:
```bash
python LPRNet_pruning.py --pretrained_model <path_to_model.pth> --pruning_ratio 0.5 --verbose True --save <output_path.pth>
```

#### Arguments:
- `--pretrained_model`: Path to the pretrained LPRNet model. Default: `weights/Final_LPRNet.pth`.
- `--pruning_ratio`: The fraction of weights to prune in each applicable layer. Default: `0.5`.
- `--verbose`: If `True`, prints the number of non-zero parameters and parameter reduction percentage. Default: `False`.
- `--save`: Path to save the pruned model. Default: `unstructured_pruned_LPRNet_model_0.5.pth`.

---

## Key Features

- **Customizable Pruning Ratio:**  
  Specify the percentage of weights to prune in applicable layers.

- **Size Reduction Analysis:**  
  Compare the size of the original and pruned models, and display the percentage reduction.

- **Modularity:**  
  The script is designed to be modular, enabling integration into larger projects.

---

## Example

To prune 30% of the weights in the model and save the pruned version:
```bash
python LPRNet_pruning.py --pretrained_model ./weights/Final_LPRNet.pth --pruning_ratio 0.3 --verbose True --save ./weights/pruned_LPRNet_0.3.pth
```

Output:
```
1. size : 1.91 MB
2. Accuracy: 82.5%
3. Inference Time: 0.15 seconds
```

---

## Notes

- Ensure the pretrained model path is correct.
- Use `--verbose` to get detailed statistics about parameter reduction.
- The script removes pruning reparameterizations before saving the pruned model, ensuring compatibility for further use.

--- 

This README serves as a reference to understand and execute the unstructured pruning process for the LPRNet model efficiently.

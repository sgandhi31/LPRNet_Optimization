# MLC Optimization Scripts

This directory contains supporting scripts for optimizing a PyTorch model using Apache TVM's auto-tuning and compilation tools. These scripts handle tasks such as model conversion, auto-tuning, and compilation, forming an essential part of the overall MLC pipeline.

---

## Scripts Overview

### 1. **`build_autotune.py`**

This script is responsible for compiling the PyTorch model into an optimized format using TVM's Relay IR and applying the best tuning results from a previously generated log file. It also benchmarks the model's inference performance.

#### Key Components:
- **Dependencies:**  
  Imports necessary modules such as `tvm`, `relay`, `autotvm`, `auto_scheduler`, and `graph_executor`.
  
- **Main Functionality:**  
  - Loads a pre-tuned model from the provided log file.
  - Converts the PyTorch model to Relay IR.
  - Compiles the model with the best optimization history.
  - Creates a TVM graph executor for benchmarking.
  - Benchmarks the inference time of the optimized model.

- **Entry Function:**  
  `compile(model_path, log_file_path, target='llvm')`  
  Takes the path to the model, the tuning log file, and the hardware target (default: `llvm`).

---

### 2. **`load_autotune.py`**

This script supports converting the PyTorch model into TVM's Relay IR, extracting auto-tuning tasks, and performing auto-tuning to optimize model performance.

#### Key Components:
- **Dependencies:**  
  Imports modules such as `torch`, `tvm`, `relay`, and `auto_scheduler`.

- **Main Functions:**
  - `convert_to_relay(model_path)`  
    Converts a PyTorch model into Relay IR, using the `LPRNet` model as an example.
    
  - `extract_tasks(model_path, target)`  
    Extracts computational tasks from the Relay IR for auto-tuning.

  - `autotune(model_path, log_file_path, target='llvm', trials=200)`  
    Tunes the model for the specified target, saves the tuning logs to the provided path, and allows configuration of the number of tuning trials.

---

## How They Work Together

1. **Auto-tuning (`load_autotune.py`):**
   - Convert the PyTorch model to TVM's Relay IR.
   - Extract tasks for the target hardware and tune them using TVM's auto-scheduler.

2. **Compilation and Benchmarking (`build_autotune.py`):**
   - Use the tuning log generated from the auto-tuning phase.
   - Compile the optimized Relay IR model.
   - Benchmark the inference performance of the tuned model.


import numpy as np
import torch
import tvm
from tvm import relay, autotvm, auto_scheduler
from mlc.load_autotune import convert_to_relay
from tvm.contrib import graph_executor, relay_viz
# Compile with the history best

def compile(model_path, log_file_path, target = 'llvm'):
  print("Compile...")
  input_shape = (1, 3, 24, 94)
  mod, params = convert_to_relay(model_path)
  with auto_scheduler.ApplyHistoryBest(log_file_path):
      with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
          lib = relay.build(mod, target=target, params=params)

  # Create graph executor
  dev = tvm.device(str(target), 0)
  module = graph_executor.GraphModule(lib["default"](dev))
  data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype("float32"))
  module.set_input("data", data_tvm)

  # Evaluate
  print("Evaluate inference time cost...")
  print(module.benchmark(dev, repeat=3, min_repeat_ms=500))
  return module

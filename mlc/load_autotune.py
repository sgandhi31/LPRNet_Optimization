import torch
import tvm
from tvm import relay, autotvm, auto_scheduler
from model.LPRNet import build_lprnet

def convert_to_relay(model_path):
  lprnet = build_lprnet(lpr_max_len=8, phase=False, class_num=68, dropout_rate=0.5)
  lprnet.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))
  lprnet.eval()

  # Define an example input
  input_shape = (1, 3, 24, 94)
  input_data = torch.randn(input_shape)

  # Convert the model to a TorchScript module
  scripted_model = torch.jit.trace(lprnet, input_data)

  # Define the input shapes for TVM
  input_name = "data"
  shape_list = [(input_name, input_shape)]

  # Convert PyTorch model to Relay (high-level IR in TVM)
  mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
  return mod, params

def extract_tasks(model_path, target):
  print("Extract tasks...")
  mod,params = convert_to_relay(model_path)
  tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
  return tasks, task_weights



def autotune(model_path, log_file_path, target = 'llvm', trials = 200):
  
  # for idx, task in enumerate(tasks):
  #     print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
  #     print(task.compute_dag)

  tasks,task_weights = extract_tasks(model_path, target)
  tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
  tune_option = auto_scheduler.TuningOptions(
      num_measure_trials= trials,
      runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
      measure_callbacks=[auto_scheduler.RecordToFile(log_file_path)],
  )

  tuner.tune(tune_option)         

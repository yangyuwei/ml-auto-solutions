# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A DAG to run end-to-end MaxText tests."""


import datetime
from airflow import models
from dags import composer_env, test_owner
from dags.vm_resource import TpuVersion, CpuVersion, Zone, DockerImage, GpuVersion, ClusterName
from dags.multipod.configs import gke_config
from airflow.utils.task_group import TaskGroup
from xlml.utils import name_format

# Run once a day at 4 am UTC (8 pm PST)
SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="maxtext_end_to_end",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "maxtext", "stable", "nightly"],
    start_date=datetime.datetime(2024, 1, 19),
    catchup=False,
) as dag:
  test_name_prefix = "maxtext"
  test_models_tpu = {
      "llama2-7b": "tpu/llama2/7b/test_llama2_7b",
      "mistral": "tpu/test_mistral",
      "gemma-2b": "tpu/gemma/2b/test_gemma",
      "gpt3": "tpu/test_gpt3",
  }

  timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
  train_base = (
      "XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 TF_FORCE_GPU_ALLOW_GROWTH=true "
      "python3 MaxText/train.py MaxText/configs/base.yml "
      "base_output_directory=gs://runner-maxtext-logs dataset_path=gs://maxtext-dataset "
      "steps=2 enable_checkpointing=false attention=dot_product"
  )
  decode_base = (
      "XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 TF_FORCE_GPU_ALLOW_GROWTH=true "
      "python3 MaxText/decode.py MaxText/configs/base.yml "
      "base_output_directory=gs://runner-maxtext-logs dataset_path=gs://maxtext-dataset "
      "steps=2 enable_checkpointing=false attention=dot_product "
      "max_target_length=128 per_device_batch_size=1"
  )
  test_models_gpu = {
      "llama2-1node": ("bash MaxText/configs/a3/llama_2_7b/1vm.sh", 1),
      "llama2-2node": ("bash MaxText/configs/a3/llama_2_7b/2vm.sh", 2),
      "llama2-4node": ("bash MaxText/configs/a3/llama_2_7b/4vm.sh", 4),
      "llama2-8node": ("bash MaxText/configs/a3/llama_2_7b/8vm.sh", 8),
      "llama2-16node": ("bash MaxText/configs/a3/llama_2_7b/16vm.sh", 16),
      "llama2-32node": ("bash MaxText/configs/a3/llama_2_7b/32vm.sh", 32),
      "llama2-64node": ("bash MaxText/configs/a3/llama_2_7b/64vm.sh", 64),
      "llama2-128node": ("bash MaxText/configs/a3/llama_2_7b/128vm.sh", 128),
  }

  for model, (test_script, nnodes) in test_models_gpu.items():
    stable_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
        accelerator_type=GpuVersion.XPK_H100_MEGA,
        gpu_zone=Zone.US_EAST4_A.value,
        time_out_in_min=720,
        test_name=f"{test_name_prefix}-stable-{model}",
        run_model_cmds=(test_script,),
        num_slices=nnodes,
        cluster_name=ClusterName.A3PLUS_CLUSTER.value,
        docker_image="gcr.io/supercomputer-testing/yangyuwei/maxtext-fastrak:06-12-2024", # a docker image for test purpose
        base_output_directory="gs://maxtext-experiments-multipod",
        test_owner=test_owner.NINA_C,
    ).run_with_run_name_generation()
    stable_gpu


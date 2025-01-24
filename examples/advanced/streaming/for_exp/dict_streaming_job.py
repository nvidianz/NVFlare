# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import tracemalloc
from torch.utils.tensorboard import SummaryWriter

from src.streaming_controller import StreamingController
from src.streaming_executor import StreamingExecutor

from nvflare import FedJob
from nvflare.app_common.streamers.container_retriever import ContainerRetriever

def main():
    # Create the FedJob
    job = FedJob(name="dict_streaming", min_clients=1)

    # Define dict_retriever component and send to both server and clients
    dict_retriever = ContainerRetriever()
    job.to_server(dict_retriever, id="dict_retriever")
    job.to_clients(dict_retriever, id="dict_retriever")

    # Define the controller workflow and send to server
    controller = StreamingController(
        dict_retriever_id="dict_retriever"
    )
    job.to_server(controller)

    # Define the executor and send to clients
    executor = StreamingExecutor(
        dict_retriever_id="dict_retriever"
    )
    job.to_clients(executor, tasks=["*"])

    # Export the job
    job_dir="workspace/jobs/dict_streaming"
    print("job_dir=", job_dir)
    job.export_job(job_dir)

    # Run the job
    work_dir="workspace/works/dict_streaming"
    print("workspace_dir=", work_dir)

    # starting the monitoring
    job.simulator_run(work_dir, n_clients=1, threads=1)

if __name__ == "__main__":
    main()

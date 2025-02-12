# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from src.df_statistics import DFStatistics

from nvflare.job_config.stats_job import StatsJob


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_clients", type=int, default=3)
    parser.add_argument("-d", "--data_root_dir", type=str, nargs="?", default="/tmp/nvflare/df_stats/data")
    parser.add_argument("-o", "--stats_output_path", type=str, nargs="?", default="statistics/adult_stats.json")
    parser.add_argument("-j", "--job_dir", type=str, nargs="?", default="/tmp/nvflare/jobs/stats_df")
    parser.add_argument("-w", "--work_dir", type=str, nargs="?", default="/tmp/nvflare/jobs/stats_df")

    return parser.parse_args()


def main():
    args = define_parser()

    n_clients = args.n_clients
    data_root_dir = args.data_root_dir
    output_path = args.stats_output_path
    job_dir = args.job_dir
    work_dir = args.work_dir

    statistic_configs = {
        "count": {},
        "mean": {},
        "sum": {},
        "stddev": {},
        "histogram": {"*": {"bins": 20}},
        "Age": {"bins": 20, "range": [0, 10]},
        "percentile": {"*": [25, 50, 75], "Age": [50, 95]},
    }
    # define local stats generator
    df_stats_generator = DFStatistics(data_root_dir=data_root_dir)

    job = StatsJob(
        job_name="stats_df",
        statistic_configs=statistic_configs,
        stats_generator=df_stats_generator,
        output_path=output_path,
    )

    sites = [f"site-{i + 1}" for i in range(n_clients)]
    job.setup_clients(sites)

    job.export_job(job_dir)

    job.simulator_run(work_dir)


if __name__ == "__main__":
    main()

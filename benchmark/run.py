import numpy as np
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", nargs="+", required=True)
parser.add_argument("-d", "--dataset", nargs="+", required=True)
parser.add_argument("-n", "--num_workers", type=int, default=2)
parser.add_argument("-r", "--run", type=int, default=0)
args = vars(parser.parse_args())


def read_cmds(model, dataset):
    pre_cmds = []
    running_cmds = []
    cmd_file = f"./benchmarking_scripts/{dataset}/{model}.sh"
    with open(cmd_file) as fr:
        for line in fr:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export"):
                pre_cmds.append(line)
            else:
                running_cmds.append(line)
    return pre_cmds, running_cmds


if __name__ == "__main__":
    total_gpu = 8
    cmd_logs = []
    model = args["model"]
    dataset = args["dataset"]
    num_workers = args["num_workers"]

    running_cmds = []
    for m in model:
        for ds in dataset:
            pre_cmds, running_cmd = read_cmds(m, ds)
            running_cmds.extend(running_cmd)

    for idx, cmd_list in enumerate(np.array_split(running_cmds, num_workers)):
        gpu_idx = idx % total_gpu
        merged_cmd = (
            "("
            + " && ".join(
                [f"CUDA_VISIBLE_DEVICES={gpu_idx} {item}" for item in cmd_list]
            )
            + ")"
        )
        merged_cmd += (
            f" > logs/{'-'.join(model)}.{'-'.join(dataset)}.multi_{idx}.log 2>&1 &"
        )
        print(merged_cmd)
        cmd_logs.append(merged_cmd)
        if args["run"] > 0:
            subprocess.Popen(merged_cmd, shell=True)

    with open("cmd_history.txt", "w") as fw:
        for item in cmd_logs:
            fw.write(item + "\n")

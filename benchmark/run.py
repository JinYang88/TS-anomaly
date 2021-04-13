import numpy as np
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required=True)
parser.add_argument("-d", "--dataset", required=True)
parser.add_argument("-n", "--num_workers", default=2)
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
    model = args["model"]
    dataset = args["dataset"]
    num_workers = args["num_workers"]

    pre_cmds, running_cmds = read_cmds(model, dataset)

    for idx, cmd_list in enumerate(np.array_split(running_cmds, num_workers)):
        merged_cmd = "(" + " && ".join([f"{item}" for item in cmd_list]) + ")"
        merged_cmd += f" > logs/{model}.{dataset}.multi_{idx}.log 2>&1 &"
        subprocess.check_output(merged_cmd, shell=True)
        print(merged_cmd)

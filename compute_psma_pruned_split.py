import numpy as np
import seaborn as sns
import json
import os

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--final_split", type=str, help="Path to the final_split.json of autoPET III dataset")
parser.add_argument("--target_dir", type=str, help="Target directory in which to save the psma-pruned split")
parser.add_argument("--n", type=int, help="n-th percentile up to which the data should be pruned")

def main(args):
    final_split = args.final_split
    target_dir = args.target_dir
    n = args.n

    with open(final_split,"r") as f:
        split = json.load(f)
    imgs = np.array(split[0]["train"] + split[0]["val"])
    
    with open("baseline_model_performance.txt","r") as f:
        all_metrics_raw = f.readlines()

    all_metrics = [
        {k:float(v) for k,v in zip(x.strip().split("-")[::2],x.strip().split("-")[1::2])} for x in all_metrics_raw
        ]
    for x in all_metrics:
        img_name = imgs[int(x["Bindex"])]
        x.update({"Img":img_name})
        x.update({"FDG":img_name.startswith("fdg")})
        x.update({"PSMA":img_name.startswith("psma")})
    
    sorted_metrics = sorted(all_metrics,key=lambda x: x["Loss"])[::-1]
    loss_cutoff = np.percentile([x["Loss"] for x in sorted_metrics if x["PSMA"]],n)
    to_be_removed_from_psma = [x["Img"] for x in sorted_metrics if x["PSMA"] if x["Loss"]<=loss_cutoff]

    with open(os.path.join(target_dir,f"cust_split_psma_{n}percent.json"),"w") as f:
        json.dump(
            [{"train": [x for x in imgs if x not in to_be_removed_from_psma],
            "val": []
            }],
            f)

if __name__ == '__main__':
    main(parser.parse_args())
"""Plot participant-averaged curves from scores.csv, retaining dataset panels."""
import argparse,csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument("scores",nargs="+")
parser.add_argument("--out",required=True)
args=parser.parse_args()
groups=defaultdict(list)
for path in args.scores:
    with open(path) as f:
        for r in csv.DictReader(f):
            groups[(r["dataset"],r["method"],r["subject"],int(r["budget"]))].append(float(r["balanced_accuracy"]))
datasets=sorted(set(k[0] for k in groups))
fig,axes=plt.subplots(len(datasets),1,figsize=(9,5*len(datasets)),squeeze=False)
for ax,dataset in zip(axes[:,0],datasets):
    methods=sorted(set(k[1] for k in groups if k[0]==dataset))
    for method in methods:
        means=[]
        for budget in (0,4,8,16,32):
            values=[np.mean(v) for k,v in groups.items() if k[0]==dataset and k[1]==method and k[3]==budget]
            means.append(np.mean(values))
        ax.plot([0,4,8,16,32],means,marker="o",label=method)
    ax.set(xlabel="Labeled target-session prefix trials",ylabel="Balanced accuracy",ylim=(0,1),
           title=dataset + (" | SOFTWARE FIXTURE ONLY" if dataset=="SYNTHETIC" else ""))
    ax.legend(fontsize=7,ncol=2); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(args.out,dpi=180)

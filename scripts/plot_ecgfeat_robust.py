#!/usr/bin/env python3
"""Export static research figures from the saved aggregate results."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]/"ecgfeat_robust_20260922"
x=json.loads((root/"analysis.json").read_text())
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout="constrained")
colors=["#7b8794","#087e8b"]
for method,label,color in zip(["default","qrs_adaptive_consensus"],["Baseline","Adaptive QRS"],colors):
    rows=sorted([r for r in x["nstdb"] if r["method"]==method and r["phase"]=="noisy"],key=lambda r:r["snr"],reverse=True)
    axes[0].plot([r["snr"] for r in rows],[r["f1"]*100 for r in rows],"o-",label=label,color=color)
axes[0].set(title="NSTDB: noisy phases only",xlabel="SNR (dB)",ylabel="QRS F1 (%)",ylim=(0,104))
axes[0].invert_xaxis();axes[0].set_xticks([24,18,12,6,0,-6]);axes[0].legend(frameon=False)
for i,(method,label,color) in enumerate(zip(["default","limited_robust"],["Baseline","Limited + refinements"],colors)):
    rows=[next(r for r in x["but"]["all"] if r["variant"]==method and r["wave"]=="P" and r["lead"]=="II" and r["tolerance_ms"]==t) for t in [50,150]]
    bars=axes[1].bar(np.arange(2)+(i-.5)*.34,[r["f1"]*100 for r in rows],width=.34,label=label,color=color)
    axes[1].bar_label(bars,fmt="%.2f",padding=3)
axes[1].set(title="BUT: native P candidates, all 50 records",xticks=[0,1],xticklabels=["50 ms","150 ms"],ylabel="P F1 (%)",ylim=(0,104))
axes[1].legend(frameon=False,loc="lower right")
activities=["sitting","maths","walking","hand_bike","jogging"]
for i,(method,label,color) in enumerate(zip(["default","adaptive"],["Baseline","Adaptive QRS"],colors)):
    rows=[next(r for r in x["gudb"]["metrics"] if r["method"]==method and r["lead"]=="cables" and r["protocol"]=="peak_75ms" and r["activity"]==a) for a in activities]
    axes[2].bar(np.arange(5)+(i-.5)*.36,[r["f1"]*100 for r in rows],width=.36,label=label,color=color)
axes[2].set(title="GUDB: cable ECG, frozen detector",ylabel="QRS F1 at 75 ms (%)",ylim=(0,104),xticks=range(5),xticklabels=["Sit","Maths","Walk","Bike","Jog"])
axes[2].legend(frameon=False,loc="lower right")
for ax in axes:ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
fig.savefig(root/"validation_results.png",dpi=170)
fig.savefig(root/"validation_results.svg")

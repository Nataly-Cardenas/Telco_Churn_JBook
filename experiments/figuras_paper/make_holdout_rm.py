import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from pypdf import PdfReader
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["DejaVu Sans"],"font.size":9,"axes.labelsize":9.5,
 "xtick.labelsize":8.5,"ytick.labelsize":8.5,"legend.fontsize":8,"lines.linewidth":1.6,"axes.grid":True,"grid.alpha":0.3,
 "pdf.fonttype":42,"savefig.bbox":"tight","axes.spines.top":False,"axes.spines.right":False})
OUT=os.environ["FIG_OUT"]; D="data"
o=pd.read_csv(f"{D}/holdout_probs_reconstruida.csv"); y=o.y.values
m=pd.read_csv(f"{D}/metricas_reconstruidas.csv").set_index("modelo")
names={"RegionalMAPEM":"RegionalMAPEM","LR_splines":"Spline logistic","HistGB":"HistGB","LightGBM":"LightGBM"}
cols=["#1F3A5F","#E0A526","#2E8B57","#7A7A7A"]; ls=["-","--","-.",":"]
fig,ax=plt.subplots(1,3,figsize=(7.4,2.6),gridspec_kw={"wspace":0.45,"width_ratios":[1,1,0.8]})
for (k,n),c,l in zip(names.items(),cols,ls):
    f,t,_=roc_curve(y,o[k]); pr,rc,_=precision_recall_curve(y,o[k])
    ax[0].plot(f,t,c=c,ls=l,label=n); ax[1].plot(rc,pr,c=c,ls=l)
ax[0].plot([0,1],[0,1],c="#BBBBBB",lw=1); ax[1].axhline(y.mean(),c="#BBBBBB",lw=1)
ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_ylim(0.2,1.02)
ax[0].legend(frameon=False,loc="lower right")
r=m.loc["RegionalMAPEM"]; cm=np.array([[r.tn,r.fp],[r.fn,r.tp]],int)
ax[2].imshow(cm,cmap="Blues"); ax[2].grid(False)
for i in range(2):
    for j in range(2): ax[2].text(j,i,f"{cm[i,j]}",ha="center",va="center",color="white" if cm[i,j]>500 else "black",fontsize=10)
ax[2].set_xticks([0,1],["Stay","Churn"]); ax[2].set_yticks([0,1],["Stay","Churn"]); ax[2].set_xlabel("Predicted"); ax[2].set_ylabel("Actual")
for a,l in zip(ax,"abc"): a.set_title(l,loc="left",fontweight="bold",fontsize=10.5)
for _ in range(6):
    fig.savefig(f"{OUT}/fig_roc_pr.pdf"); w=float(PdfReader(f"{OUT}/fig_roc_pr.pdf").pages[0].mediabox.width)/72
    if w<=6.25: break
    fw,fh=fig.get_size_inches(); fig.set_size_inches(fw-(w-6.25)-0.02,fh)
print(w)
# filas LaTeX
order=["RegionalMAPEM","LR_splines","HistGB","LightGBM","XGB","RF","CatBoost","LR"]
lab={"RegionalMAPEM":"RegionalMAPEM","LR_splines":"Spline logistic","HistGB":"HistGB","LightGBM":"LightGBM","XGB":"XGBoost","RF":"Random Forest","CatBoost":"CatBoost","LR":"Logistic regression"}
mm=m.loc[order]; cs=[("ll",False,4),("auc",True,4),("ap",True,4),("mcc",True,3),("ba",True,3),("f1",True,3)]
best={};sec={}
for c,hi,_ in cs:
    s=mm[c].sort_values(ascending=not hi); best[c],sec[c]=s.index[0],s.index[1]
def fmt(v,d): return f"{v:.{d}f}".replace(".","{,}")
for k in order:
    cells=[]
    for c,hi,d in cs:
        s="0"+fmt(mm.loc[k,c],d) if False else fmt(mm.loc[k,c],d)
        s=f"0{{,}}{s.split('{,}')[1]}" if s.startswith("0") else s
        s=r"\textbf{"+s+"}" if best[c]==k else (r"\underline{"+s+"}" if sec[c]==k else s)
        cells.append(s)
    print(f"{lab[k]} & "+" & ".join(cells)+f" & {int(mm.loc[k,'tn'])} / {int(mm.loc[k,'fp'])} & {int(mm.loc[k,'fn'])} / {int(mm.loc[k,'tp'])} \\\\")

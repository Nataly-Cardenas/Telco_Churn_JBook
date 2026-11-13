
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
OUT = os.environ.get("FIG_OUT", os.path.join(os.getcwd(), "paper", "figures"))
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Latin Modern Sans", "DejaVu Sans"],
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "mathtext.fontset": "dejavusans"})
FOCAL, FILL, ACC, ACCE = "#1F3A5F", "#EEF1F5", "#FFF4D6", "#C9962B"
FS = 8.5
fig, ax = plt.subplots(figsize=(6.3, 3.5)); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
def box(x, y, w, h, lines, fc=FILL, ec=FOCAL, title=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5", fc=fc, ec=ec, lw=1.6))
    txt = "\n".join(lines)
    if title:
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=FS, fontweight="bold")
        ax.text(x + w / 2, y + (h - 6) / 2, txt, ha="center", va="center", fontsize=FS, linespacing=1.3)
    else:
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=FS, linespacing=1.3)
def arrow(x0, y0, x1, y1, lab=None, lx=0, ly=0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=FOCAL, lw=1.6))
    if lab:
        ax.text((x0 + x1) / 2 + lx, (y0 + y1) / 2 + ly, lab, ha="center", va="center", fontsize=FS, color=FOCAL)
box(0.5, 25, 11, 10, ["$x$"], title="Input")
box(15, 37, 30, 20, ["tuned once, then frozen", "out-of-fold $p_m(x_i)$", "(3-fold cross-fitting)"], title="M calibrated experts")
box(15, 1, 30, 23, ["continuous features", "→ normal scores", "→ PCA (2 components)", "→ GMM, K regions"], title="Unsupervised regions")
box(51, 42, 24, 15, ["EM with $K=1$", "prior Dir$(1 + c\\,g)$"], title="Global weights g")
box(51, 12, 25, 22, ["region weights $\\pi_{km}$", "concave objective", "for fixed regions"], fc=ACC, ec=ACCE, title="MAP-EM")
box(81, 17, 18.5, 12, ["$\\hat p(x)$"], title="Output")
arrow(11.8, 32, 14.6, 44); arrow(11.8, 28, 14.6, 16)
arrow(45.4, 50, 50.6, 50); arrow(63, 41.6, 63, 34.4)
arrow(45.4, 42, 50.6, 30); ax.text(49.3, 39.0, "$p_m$", fontsize=FS, color=FOCAL, ha="center")
arrow(45.4, 13, 50.6, 18); ax.text(48.6, 10.8, "$r_k$", fontsize=FS, color=FOCAL, ha="center")
arrow(75.4, 23, 80.6, 23)
ax.text(76, 6.5, "$\\hat p(x)=\\sum_k r_k(x)\\sum_m \\pi_{km}\\,p_m(x)$", ha="center", va="center", fontsize=FS, color=FOCAL)
fig.savefig(os.path.join(OUT, "fig_architecture.pdf")); fig.savefig(os.path.join(OUT, "fig_architecture.png"), dpi=200)

"""docs/Reference.md(논문 형식 보고서)의 그림 1~6을 만든다.

그림 1 방법 개요, 그림 2 ROC, 그림 3 살아 있는 에피소드 AUROC, 그림 4 중단 trade-off,
그림 5 두 모델의 AUROC 비교, 그림 6 정성 분석(공개 모델, 물체 위치, 가장 일찍 잡은 실패).
입력은 evaluate.py 결과 json 두 개와 공개 모델 수집 폴더(그림 6의 chunk 그림)다.

사용 (WSL, matplotlib 있는 환경):
  python wac/make_paper_figures.py results/detection_base_full_v2.json results/detection_ft_full.json \
      data/libero_spatial docs/paper_figures
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from episode_index import task_dirs_for  # noqa: E402

BASE_JSON, FT_JSON, DATA, OUT = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
base = json.load(open(BASE_JSON, encoding="utf-8"))
ft = json.load(open(FT_JSON, encoding="utf-8"))

for f in ("/mnt/c/Windows/Fonts/malgun.ttf", "/mnt/c/Windows/Fonts/malgunbd.ttf",
          "C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf"):
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": ["Malgun Gothic", "DejaVu Sans"], "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300, "pdf.fonttype": 42,
    "legend.frameon": False,
})
# 범주 색은 고정 순서(검증한 팔레트의 1~3번 슬롯): D 파랑, A 주황, D-WAC 청록. 선 모양과 표식으로도 구분한다.
COL = {"D": "#2a78d6", "A": "#eb6834", "D-WAC": "#1baf7a"}
LS = {"D": "-", "A": "--", "D-WAC": ":"}
MK = {"D": "o", "A": "s", "D-WAC": "^"}
NAME = {"D": "D (결합)", "A": "A (가치)", "D-WAC": "D-WAC"}
COND = {"obj_0.090": "물체 위치 ±9cm", "wristroll_50": "손목 카메라 회전 ±50°"}
MODEL = {"base": "공개 모델", "ft": "파인튜닝 모델"}
T_END = 230   # 실패 에피소드는 모두 230 step(대기 10 + 최대 220)에서 끝난다


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("저장:", name)


def final_scores(r):
    s = np.array([e["R"][-1] for e in r["episodes"]], float)
    y = np.array([e["fail"] for e in r["episodes"]], int)
    return s, y


def roc(s, y):
    o = np.argsort(-s, kind="mergesort")
    s, y = s[o], y[o]
    keep = np.r_[np.diff(s) != 0, True]           # 같은 점수는 한 번에 넘긴다
    tp, fp = np.cumsum(y)[keep], np.cumsum(1 - y)[keep]
    return np.r_[0, fp / (1 - y).sum()], np.r_[0, tp / y.sum()]


# ---------------- 그림 1: 방법 개요 ----------------
def fig_overview():
    fig = plt.figure(figsize=(7.2, 3.1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 43)
    ax.axis("off")

    def box(x, y, w, h, text, edge=AXIS, fill="#f6f6f3", bold=False, size=6.8, lw=0.9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8",
                                    fc=fill, ec=edge, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
                fontweight="bold" if bold else "normal", color=INK, linespacing=1.45)

    def arrow(x1, y1, x2, y2, color=INK2, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
                                     lw=0.9, color=color, linestyle=ls, shrinkA=0, shrinkB=0))

    ax.text(1, 41, "① chunk k 추론 (step $t_k$)", fontsize=7.8, fontweight="bold")
    ax.text(49, 41, "② 결정 입력", fontsize=7.8, fontweight="bold")
    ax.text(68.5, 41, "③ 탐지와 중단", fontsize=7.8, fontweight="bold")
    ax.text(27.2, 7.5, "실행 후 비교
(step $t_k$+16)", fontsize=7.4, fontweight="bold", color=INK2, ha="right", va="center")
    y0, h = 18.5, 19.5          # 윗줄 상자의 아래 끝과 높이
    box(0.3, 23, 11.8, 10, "관측 $o_{t_k}$\n3인칭·손목 이미지\n관절 9차원")
    box(14, y0, 12, h, "Cosmos Policy\n\n2B DiT, 28블록\ndenoising 5 step", edge=INK2, lw=1.2)
    box(28.5, y0, 18, h, "행동 묶음 (16 step)\n미래 이미지·관절 $\\hat z, \\hat p$\n가치 $\\hat V$\n"
        "hidden (블록 7·14·21·28)\n표본 4개 (노이즈만 다름)")
    box(49, y0, 17, h, "$A_k = 1 - \\hat V_k$\n$F_k$ = 표본 간 미래 분산\n$H_k$ = hidden 평균\n"
        "$B_{k-1}, C_{k-1}$ = 직전 비교")
    box(68.5, y0, 16, h, "calib 성공으로 z-score\n\nD-WAC: $z_A, z_F, z_B, z_C$\nD-hidden: PCA($H_k$)\n"
        "→ stacking $s_k$", edge=COL["D-WAC"])
    box(87, y0, 12.5, h, "$R_k = \\max_{j \\leq k}\\, s_j$\n\n$R_k > \\tau$ 이면\n멈춤\n아니면 chunk k 실행",
        edge=COL["D"], fill="#eaf2fc")
    for x1, x2 in ((12.3, 13.8), (26.2, 28.3), (46.7, 48.8), (66.2, 68.3), (84.7, 86.8)):
        arrow(x1, y0 + h / 2, x2, y0 + h / 2)
    box(28.5, 2.5, 18, 10, "행동 16 step 실행\n실제 관측 $o_{t_k+16}$")
    box(49, 2.5, 17, 10, "실제 이미지를 33장 시퀀스의\n같은 자리에 넣어 VAE 인코딩\n"
        "$B = \\Vert \\hat p - p \\Vert$,  $C$ = mean$(\\hat z - z)^2$", size=6.4)
    arrow(37.5, y0 - 0.2, 37.5, 12.7)
    arrow(46.7, 7.5, 48.8, 7.5)
    arrow(57.5, 12.7, 57.5, y0 - 0.2)
    ax.text(58.3, 15.6, "다음 chunk에서 사용", fontsize=6.4, color=INK2)
    ax.text(87, 14.4, "$\\tau$: calib 성공 에피소드\n최종 점수의 95% 분위수", fontsize=6.4, color=INK2, va="top")
    save(fig, "fig1_overview")


# ---------------- 그림 2: ROC ----------------
def fig_roc():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    for ax, cond in zip(axes, COND):
        ax.plot([0, 1], [0, 1], color=AXIS, lw=0.8, zorder=1)
        for det in ("D", "D-WAC", "A"):
            s, y = final_scores(base[cond][det])
            fpr, tpr = roc(s, y)
            ax.step(fpr, tpr, where="post", color=COL[det], ls=LS[det], lw=1.6,
                    label=f"{NAME[det]}  {base[cond][det]['auroc']:.3f}", zorder=3)
        ax.set_title(COND[cond])
        ax.set_xlabel("거짓 양성률 (경고한 성공 에피소드 비율)")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_aspect("equal")
        ax.legend(loc="lower right", title="AUROC", title_fontsize=7.5, handlelength=2.6)
    axes[0].set_ylabel("참 양성률 (경고한 실패 에피소드 비율)")
    save(fig, "fig2_roc")


# ---------------- 그림 3: 살아 있는 에피소드 AUROC ----------------
def fig_alive():
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.3), sharex=True, sharey=True)
    for i, (mkey, res) in enumerate((("base", base), ("ft", ft))):
        for j, cond in enumerate(COND):
            ax = axes[i, j]
            ax.axhline(0.5, color=AXIS, lw=0.9, zorder=1)
            for det in ("D", "D-WAC", "A"):
                pts = [(x["k"], x["auroc"]) for x in res[cond][det]["alive_auroc"]
                       if x["n_fail"] >= 5 and x["n"] - x["n_fail"] >= 5]
                ks, au = zip(*pts)
                ax.plot(ks, au, color=COL[det], ls=LS[det], marker=MK[det], ms=3.6, lw=1.5,
                        mfc=COL[det], mec="white", mew=0.6, label=NAME[det], zorder=3)
            last = [x for x in res[cond]["D"]["alive_auroc"] if x["n_fail"] >= 5 and x["n"] - x["n_fail"] >= 5][-1]
            ax.text(0.98, 0.04, f"마지막 점: 성공 {last['n'] - last['n_fail']}, 실패 {last['n_fail']}",
                    transform=ax.transAxes, ha="right", fontsize=6.6, color=INK2)
            ax.set_ylim(0.15, 1.02)
            ax.set_xticks(range(1, 11))
            ax.set_xlim(0.6, 10.4)
            if i == 0:
                ax.set_title(COND[cond])
            if j == 0:
                ax.set_ylabel(f"{MODEL[mkey]}\nAUROC (살아 있는 에피소드)")
            if i == 1:
                ax.set_xlabel("chunk 순번 k  (step ≈ 10 + 16(k−1))")
    axes[0, 0].legend(loc="upper left", ncol=3, handlelength=2.6, columnspacing=1.2)
    fig.align_ylabels(axes[:, 0])
    save(fig, "fig3_alive_auroc")


# ---------------- 그림 4: 중단 trade-off ----------------
def stop_curve(r):
    eps = r["episodes"]
    vals = np.unique(np.concatenate([np.array(e["R"], float) for e in eps]))
    taus = np.r_[vals[0] - 1, vals, vals[-1] + 1]
    succ = [np.array(e["R"], float) for e in eps if not e["fail"]]
    fails = [(np.array(e["R"], float), e["t"]) for e in eps if e["fail"]]
    fa, sv = [], []
    for tau in taus:
        fa.append(np.mean([(R > tau).any() for R in succ]))
        saved = 0.0
        for R, t in fails:
            hit = np.nonzero(R > tau)[0]
            if len(hit):
                saved += T_END - t[hit[0]]
        sv.append(saved / (T_END * len(fails)))
    o = np.lexsort((np.array(sv), np.array(fa)))
    return np.array(fa)[o], np.array(sv)[o]


def fig_stop():
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), sharex=True, sharey=True)
    for i, (mkey, res) in enumerate((("base", base), ("ft", ft))):
        for j, cond in enumerate(COND):
            ax = axes[i, j]
            ax.axvline(0.05, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
            for det in ("D", "D-WAC", "A"):
                r = res[cond][det]
                fa, sv = stop_curve(r)
                # 같은 오중단율에서는 가장 많이 아낀 점만 남겨 위쪽 경계를 그린다
                ufa = np.unique(fa)
                usv = np.array([sv[fa == v].max() for v in ufa])
                ax.step(ufa, np.maximum.accumulate(usv), where="post", color=COL[det], ls=LS[det], lw=1.5,
                        label=NAME[det], zorder=3)
                ax.plot(r["false_alarm"], r["saved_fraction_of_failure_steps"], marker=MK[det], ms=6.5,
                        color=COL[det], mec="white", mew=1.4, ls="none", zorder=4)
            ax.set_xlim(-0.01, 1.0)
            ax.set_ylim(-0.01, 0.95)
            if i == 0:
                ax.set_title(COND[cond])
            if j == 0:
                ax.set_ylabel(f"{MODEL[mkey]}\n아낀 동작 (실패 에피소드 step 비율)")
            if i == 1:
                ax.set_xlabel("오중단율 (멈춘 성공 에피소드 비율)")
    axes[0, 0].text(0.06, 0.88, "보정 목표 5%", fontsize=6.6, color=INK2)
    axes[0, 0].legend(loc="lower right", handlelength=2.6)
    fig.align_ylabels(axes[:, 0])
    save(fig, "fig4_stop_tradeoff")


# ---------------- 그림 5: 두 모델 AUROC ----------------
def fig_models():
    dets = ["A", "D", "D-WAC", "D-hidden", "D-hidden-now", "D-hidden-future", "D-concat"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
    ypos = np.arange(len(dets))[::-1]
    for ax, cond in zip(axes, COND):
        for off, (mkey, res) in ((0.17, ("base", base)), (-0.17, ("ft", ft))):
            for y, det in zip(ypos, dets):
                r = res[cond][det]
                lo, hi = r["auroc_ci"]
                ax.plot([lo, hi], [y + off] * 2, color=INK2 if mkey == "base" else INK, lw=1.0, zorder=2)
                ax.plot(r["auroc"], y + off, marker="o" if mkey == "base" else "D", ms=4.8,
                        mfc="white" if mkey == "base" else INK, mec=INK2 if mkey == "base" else INK,
                        mew=1.1, ls="none", zorder=3, label=MODEL[mkey] if y == ypos[0] else None)
        ax.set_yticks(ypos)
        ax.set_yticklabels(dets)
        ax.set_xlim(0.45, 1.005)
        ax.set_title(COND[cond])
        ax.set_xlabel("AUROC (95% 신뢰구간)")
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower left", handletextpad=0.4)
    save(fig, "fig5_models")


# ---------------- 그림 6: 정성 분석 ----------------
def fig_qual():
    cond = "obj_0.090"
    rD, rA = base[cond]["D"], base[cond]["A"]
    dirs = task_dirs_for(rD["episodes"], DATA, cond)
    cands = [i for i, e in enumerate(rD["episodes"]) if e["fail"] and e["stop_t"] is not None]
    i = min(cands, key=lambda j: rD["episodes"][j]["stop_t"])
    eD, eA = rD["episodes"][i], rA["episodes"][i]
    assert eD["name"] == eA["name"] and eD["t"] == eA["t"]
    k_stop = eD["t"].index(eD["stop_t"])
    ep = eD["name"].split("/")[-1]
    png = os.path.join(dirs[i], f"{ep}_frames", f"q{max(k_stop - 1, 0):02d}.png")
    task = os.path.basename(dirs[i]).split("_")[0]

    fig = plt.figure(figsize=(7.0, 3.3))
    axD = fig.add_axes([0.08, 0.56, 0.50, 0.36])
    axA = fig.add_axes([0.08, 0.13, 0.50, 0.36], sharex=axD)
    for ax, r, e, det in ((axD, rD, eD, "D"), (axA, rA, eA, "A")):
        R = np.array(e["R"], float)
        ax.plot(e["t"], R, color=COL[det], ls=LS[det], marker=MK[det], ms=3.8, lw=1.5, mec="white", mew=0.6)
        ax.axhline(r["threshold"], color=INK2, lw=0.9, ls=(0, (3, 2)))
        ax.text(222, r["threshold"], " τ", va="center", fontsize=7.5, color=INK2)
        ax.axvline(eD["stop_t"], color=COL["D"], lw=1.0, alpha=0.5)
        ax.set_ylabel(f"{NAME[det]}\n점수 $R_k$")
        ax.set_xlim(0, 232)
    lo, hi = axD.get_ylim()
    axD.text(eD["stop_t"] + 4, lo + 0.18 * (hi - lo), f"D가 멈춤 (step {eD['stop_t']})", fontsize=7, color=INK)
    axA.text(eD["stop_t"] + 3, axA.get_ylim()[0] + 0.12 * np.ptp(axA.get_ylim()),
             "A는 끝까지 τ를 넘지 않음" if eA["stop_t"] is None else f"A는 step {eA['stop_t']}에 멈춤",
             fontsize=7, color=INK)
    plt.setp(axD.get_xticklabels(), visible=False)
    axA.set_xlabel("step")
    init = ep.split("_")[0].replace("init", "")
    axD.set_title(f"공개 모델, {COND[cond]}, 지시문 {task[1:].lstrip('0')}, 초기 상태 {init} (실패)", loc="left")
    axI = fig.add_axes([0.61, 0.04, 0.38, 0.86])
    axI.imshow(Image.open(png))
    axI.axis("off")
    axI.set_title("위부터 현재, 예측한 16 step 뒤, 실제 16 step 뒤\n(왼쪽 3인칭, 오른쪽 손목)", fontsize=7, color=INK2)
    save(fig, "fig6_qualitative")
    print("그림 6 에피소드:", dirs[i], ep, "chunk 그림", png)


fig_overview()
fig_roc()
fig_alive()
fig_stop()
fig_models()
fig_qual()

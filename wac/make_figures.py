"""보고서용 정성 분석 그림. evaluate.py 결과 json과 수집 데이터의 chunk 그림으로 만든다.

조건마다 세 에피소드를 고른다: 일찍 잡은 실패, 잘못 멈춘 성공, 놓친 실패.
그림마다 D와 A의 점수 곡선을 같은 시간축을 쓰는 두 패널에 그리고, 아래에 chunk의 '현재 | 예측 | 실제' 그림을 붙인다.

사용: python wac/make_figures.py <결과 json> <데이터 폴더> <출력 폴더> [--det D]
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("result")
p.add_argument("data_dir")
p.add_argument("out_dir")
p.add_argument("--det", default="D")
a = p.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
res = json.load(open(a.result, encoding="utf-8"))


def frame_png(name, k):
    cond, split, task_dir, ep = name.split("/")
    path = os.path.join(a.data_dir, cond, split, task_dir, ep + "_frames", f"q{k:02d}.png")
    return path if os.path.isfile(path) else None


def check_episode_order(cond, detector_eps, value_eps):
    seen = set()
    for e, b in zip(detector_eps, value_eps, strict=True):
        name = e["name"]
        parts = name.split("/")
        if len(parts) != 4 or parts[0] != cond:
            raise ValueError(f"과제 폴더가 포함된 에피소드 이름이 필요합니다: {name}")
        if name in seen:
            raise ValueError(f"중복된 에피소드 이름: {name}")
        if (name, e["t"], e["fail"]) != (b["name"], b["t"], b["fail"]):
            raise ValueError(f"탐지기별 에피소드 기록이 일치하지 않습니다: {name}, {b['name']}")
        seen.add(name)


def pick(eps):
    det_fail = [i for i, e in enumerate(eps) if e["fail"] and e["stop_t"] is not None]
    false_stop = [i for i, e in enumerate(eps) if not e["fail"] and e["stop_t"] is not None]
    missed = [i for i, e in enumerate(eps) if e["fail"] and e["stop_t"] is None]
    out = {}
    if det_fail:
        out["early_detect"] = min(det_fail, key=lambda i: eps[i]["stop_t"])
    if false_stop:
        out["false_stop"] = false_stop[0]
    if missed:
        out["missed"] = missed[0]
    return out


def plot_scores(ax, detector, episode, det, style, color):
    t = episode["t"]
    threshold_k = detector["threshold_k"]
    ax.plot(t, episode["R"], style, color=color, label="점수 누적 최댓값")
    ax.step(t, [threshold_k[min(i, len(threshold_k) - 1)] for i in range(len(t))],
            "--", where="post", label="시점별 임계값")
    ax.axhline(detector["threshold"], color="gray", lw=0.8, label="단일 임계값")
    if episode["stop_t"] is not None:
        ax.axvline(episode["stop_t"], color="red", lw=1.2, label=f"중단 (step {episode['stop_t']})")
    ax.set_ylabel(f"{det} 점수")
    ax.legend(fontsize=7, loc="upper left")


TITLE = {"early_detect": "일찍 잡은 실패", "false_stop": "잘못 멈춘 성공", "missed": "놓친 실패"}
for f in ("/mnt/c/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgun.ttf"):   # WSL에서 Windows 한글 글꼴을 쓴다
    if os.path.exists(f):
        matplotlib.font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
for cond, r in res.items():
    if a.det not in r:
        continue
    d, base = r[a.det], r["A"]
    check_episode_order(cond, d["episodes"], base["episodes"])
    for kind, i in pick(d["episodes"]).items():
        e, b = d["episodes"][i], base["episodes"][i]
        k_stop = e["t"].index(e["stop_t"]) if e["stop_t"] is not None else None
        # 중단 판단에 쓰는 예측 오차는 직전 chunk에서 계산하므로 그 chunk의 그림을 붙인다.
        png = frame_png(e["name"], max(k_stop - 1, 0) if k_stop is not None else len(e["t"]) - 1)
        fig = plt.figure(figsize=(9, 8 if png else 5), constrained_layout=True)
        grid = fig.add_gridspec(3 if png else 2, 1, height_ratios=[1, 1, 2] if png else None)
        ax_d = fig.add_subplot(grid[0])
        ax_a = fig.add_subplot(grid[1], sharex=ax_d)
        plot_scores(ax_d, d, e, a.det, "o-", "tab:blue")
        plot_scores(ax_a, base, b, "A", "s:", "tab:green")
        ax_d.tick_params(labelbottom=False)
        ax_a.set_xlabel("step")
        ax_d.set_title(f"{e['name']} / {'실패' if e['fail'] else '성공'} / {TITLE[kind]}")
        if png:
            ax_img = fig.add_subplot(grid[2])
            with Image.open(png) as img:
                ax_img.imshow(img)
            ax_img.axis("off")
        out = os.path.join(a.out_dir, f"{cond}_{kind}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print("저장:", out)

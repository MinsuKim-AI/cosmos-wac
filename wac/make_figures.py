"""보고서용 정성 분석 그림. evaluate.py 결과 json과 수집 데이터의 chunk 그림으로 만든다.

조건마다 세 에피소드를 고른다: 일찍 잡은 실패, 잘못 멈춘 성공, 놓친 실패.
그림 하나 = 위: 탐지기 D와 가치 A의 점수 곡선과 임계값, 중단 시점 / 아래: 중단한 chunk의 '현재 | 예측 | 실제' 그림.

사용: python wac/make_figures.py <결과 json> <데이터 폴더> <출력 폴더> [--det D]
"""

import argparse
import glob
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
    cond, split, ep = name.split("/")
    hits = glob.glob(os.path.join(a.data_dir, cond, split, "D*", ep + "_frames", f"q{k:02d}.png"))
    return hits[0] if hits else None


def pick(eps):
    det_fail = [e for e in eps if e["fail"] and e["stop_t"] is not None]
    false_stop = [e for e in eps if not e["fail"] and e["stop_t"] is not None]
    missed = [e for e in eps if e["fail"] and e["stop_t"] is None]
    out = {}
    if det_fail:
        out["early_detect"] = min(det_fail, key=lambda e: e["stop_t"])
    if false_stop:
        out["false_stop"] = false_stop[0]
    if missed:
        out["missed"] = missed[0]
    return out


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
    by_name = {e["name"]: e for e in base["episodes"]}
    for kind, e in pick(d["episodes"]).items():
        k_stop = e["t"].index(e["stop_t"]) if e["stop_t"] is not None else None
        # 중단 판단에 쓴 관절·이미지 차이는 직전 chunk의 예측과 실제이므로 그 chunk 그림을 붙인다
        png = frame_png(e["name"], max(k_stop - 1, 0) if k_stop is not None else len(e["t"]) - 1)
        fig = plt.figure(figsize=(9, 6.5))
        ax = fig.add_axes([0.08, 0.58, 0.88, 0.36])
        ax.plot(e["t"], e["R"], "o-", label=f"탐지기 {a.det} (점수 누적 최댓값)")
        ax.step(e["t"], [d["threshold_k"][min(i, len(d["threshold_k"]) - 1)] for i in range(len(e["t"]))],
                "--", where="post", label="시점별 임계값")
        ax.axhline(d["threshold"], color="gray", lw=0.8, label="단일 임계값")
        if e["stop_t"] is not None:
            ax.axvline(e["stop_t"], color="red", lw=1.2, label=f"중단 (step {e['stop_t']})")
        ax2 = ax.twinx()
        b = by_name.get(e["name"])
        if b:
            ax2.plot(b["t"], b["R"], "s:", color="tab:green", alpha=0.7, label="가치 A (z 누적 최댓값)")
            ax2.set_ylabel("A 점수")
        ax.set_xlabel("step")
        ax.set_ylabel(f"{a.det} 점수")
        ax.set_title(f"{cond} / {e['name'].split('/')[-1]} / {'실패' if e['fail'] else '성공'} / {TITLE[kind]}")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
        if png:
            ax3 = fig.add_axes([0.02, 0.02, 0.96, 0.48])
            ax3.imshow(Image.open(png))
            ax3.axis("off")
        out = os.path.join(a.out_dir, f"{cond}_{kind}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print("저장:", out)

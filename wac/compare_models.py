"""기본 모델과 파인튜닝 모델의 탐지 결과(evaluate.py 출력 json)를 비교하는 표 (H3).

에피소드 점수는 마지막 chunk의 누적 최대 점수. 두 모델은 rollout이 달라 짝이 맞지 않으므로
AUROC 차이의 95% 구간은 모델별로 에피소드를 따로 다시 뽑는 bootstrap(2000번)으로 구한다.
사용: python wac/compare_models.py <기본 json> <파인튜닝 json> <출력 md>
"""
import json
import sys

import numpy as np


def auroc(s, y):
    s, y = np.asarray(s), np.asarray(y)
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return np.nan
    return ((p[:, None] > n[None]).mean() + 0.5 * (p[:, None] == n[None]).mean())


def ep_scores(r):
    return np.array([e["R"][-1] if e["R"] else 0.0 for e in r["episodes"]]), np.array([e["fail"] for e in r["episodes"]])


base, ft = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
rng = np.random.default_rng(0)
out = ["# H3: 기본 모델 vs 파인튜닝 모델", ""]
for cond in base:
    if cond not in ft:
        continue
    b, f = base[cond], ft[cond]
    kinds = [k for k in b if not k.startswith("_") and k in f]
    sb, yb = ep_scores(b["A"]); sf, yf = ep_scores(f["A"])
    out += [f"## {cond}", "",
            f"평가 에피소드: 기본 {len(yb)}개(실패 {yb.sum()}), 파인튜닝 {len(yf)}개(실패 {yf.sum()}). "
            f"성공률 기본 {b['A']['success_rate_before']:.2f}, 파인튜닝 {f['A']['success_rate_before']:.2f}", "",
            "| 탐지기 | 기본 AUROC | 파인튜닝 AUROC | 차이 [95%] | 기본 오경보/탐지/아낀 동작 | 파인튜닝 오경보/탐지/아낀 동작 |",
            "|---|---|---|---|---|---|"]
    for k in kinds:
        d = [np.nan]
        if "episodes" in b[k] and "episodes" in f[k]:
            sb, yb = ep_scores(b[k]); sf, yf = ep_scores(f[k]); d = []
        for _ in range(2000 if len(d) == 0 else 0):
            ib = rng.integers(0, len(yb), len(yb)); jf = rng.integers(0, len(yf), len(yf))
            d.append(auroc(sf[jf], yf[jf]) - auroc(sb[ib], yb[ib]))
        lo, hi = np.nanpercentile(d, [2.5, 97.5]) if np.isfinite(d).any() else (np.nan, np.nan)
        st = lambda r: f"{r['false_alarm']:.2f}/{r['detection_rate']:.2f}/{r['saved_fraction_of_failure_steps']:.2f}"
        out.append(f"| {k} | {b[k]['auroc']:.3f} | {f[k]['auroc']:.3f} | {f[k]['auroc'] - b[k]['auroc']:+.3f} [{lo:+.3f}, {hi:+.3f}] | {st(b[k])} | {st(f[k])} |")
    out.append("")
open(sys.argv[3], "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))

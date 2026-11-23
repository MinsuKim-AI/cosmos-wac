"""기본 모델과 파인튜닝 모델의 탐지 결과(evaluate.py 출력 json)를 비교하는 표 (H3).

에피소드 점수는 마지막 chunk의 누적 최대 점수. AUROC 차이의 95% 구간은 독립 표본과
대응 시나리오를 각각 2000번 재표집해 구한다. 대응 키는 조건, 지시문, 초기 상태이며
성공·실패 라벨은 각 모델의 값을 유지한다.
사용: python wac/compare_models.py <기본 json> <파인튜닝 json> <출력 md>
"""
import json
import sys

import numpy as np


BOOT_REPS = 2000
LEGACY_SCORE_DETECTORS = ("A", "D", "D-WAC")


def auroc(s, y):
    s, y = np.asarray(s), np.asarray(y)
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return np.nan
    return ((p[:, None] > n[None]).mean() + 0.5 * (p[:, None] == n[None]).mean())


def ep_scores(r):
    episodes = r["episodes"]
    return (np.array([e["R"][-1] if e["R"] else 0.0 for e in episodes]),
            np.array([e["fail"] for e in episodes]))


def scenario_indices(cond, episodes):
    # 기존 결과에는 이름에 지시문 폴더가 빠져 있고 별도 필드도 없다.
    if not all("task" in e and "init" in e for e in episodes):
        return None
    names = set()
    indices = {}
    for i, e in enumerate(episodes):
        name = e["name"]
        parts = name.split("/")
        if len(parts) != 4 or parts[0] != cond or name in names:
            raise ValueError(f"에피소드 이름이 형식에 맞지 않거나 중복됨: {name}")
        scenario = (cond, e["task"], e["init"])
        if scenario in indices:
            raise ValueError(f"시나리오 키 중복: {scenario}")
        names.add(name)
        indices[scenario] = i
    return indices


def paired_difference(cond, b, f, sb, yb, sf, yf, rng):
    base_indices = scenario_indices(cond, b["episodes"])
    ft_indices = scenario_indices(cond, f["episodes"])
    if base_indices is None or ft_indices is None:
        return "—", "—"
    shared = [key for key in base_indices if key in ft_indices]
    n = len(shared)
    if not n:
        return n, "—"
    ib = [base_indices[key] for key in shared]
    jf = [ft_indices[key] for key in shared]
    sb, yb, sf, yf = sb[ib], yb[ib], sf[jf], yf[jf]
    delta = auroc(sf, yf) - auroc(sb, yb)
    if not np.isfinite(delta):
        return n, "—"
    differences = []
    for _ in range(BOOT_REPS):
        indices = rng.integers(0, n, n)
        differences.append(auroc(sf[indices], yf[indices]) - auroc(sb[indices], yb[indices]))
    return n, format_difference(delta, differences)


def format_difference(delta, differences):
    if np.isfinite(differences).any():
        lo, hi = np.nanpercentile(differences, [2.5, 97.5])
        return f"{delta:+.3f} [{lo:+.3f}, {hi:+.3f}]"
    return f"{delta:+.3f} [—]"


def stop_summary(r):
    return f"{r['false_alarm']:.2f}/{r['detection_rate']:.2f}/{r['saved_fraction_of_failure_steps']:.2f}"


with open(sys.argv[1], encoding="utf-8") as stream:
    base = json.load(stream)
with open(sys.argv[2], encoding="utf-8") as stream:
    ft = json.load(stream)
rng = np.random.default_rng(0)
paired_rng = np.random.default_rng(1)
extra_rng = np.random.default_rng(2)
out = ["# H3: 기본 모델 vs 파인튜닝 모델", "",
       "대응 분석은 두 모델에 공통으로 있는 조건·지시문·초기 상태를 함께 재표집하며, "
       "각 모델의 성공·실패 라벨을 유지한다.", ""]
for cond in base:
    if cond not in ft:
        continue
    b, f = base[cond], ft[cond]
    kinds = [k for k in b if not k.startswith("_") and k in f]
    _, yb = ep_scores(b["A"])
    _, yf = ep_scores(f["A"])
    out += [f"## {cond}", "",
            f"평가 에피소드: 기본 {len(yb)}개(실패 {yb.sum()}), 파인튜닝 {len(yf)}개(실패 {yf.sum()}). "
            f"성공률 기본 {b['A']['success_rate_before']:.2f}, 파인튜닝 {f['A']['success_rate_before']:.2f}", "",
            "| 탐지기 | 기본 AUROC | 파인튜닝 AUROC | 비대응 차이 [95% CI] | "
            "대응 시나리오 수 | 대응 차이 [95% CI] | 기본 오경보/탐지/아낀 동작 | 파인튜닝 오경보/탐지/아낀 동작 |",
            "|---|---|---|---|---|---|---|---|"]
    for k in kinds:
        base_result, ft_result = b[k], f[k]
        delta = ft_result["auroc"] - base_result["auroc"]
        differences = []
        n_paired, paired = "—", "—"
        if "episodes" in base_result and "episodes" in ft_result:
            sb, yb = ep_scores(base_result)
            sf, yf = ep_scores(ft_result)
            if len(yb) and len(yf):
                # 새로 저장한 탐지기의 표본은 기존 비대응 구간의 난수열을 바꾸지 않는다.
                unpaired_rng = rng if k in LEGACY_SCORE_DETECTORS else extra_rng
                for _ in range(BOOT_REPS):
                    ib = unpaired_rng.integers(0, len(yb), len(yb))
                    jf = unpaired_rng.integers(0, len(yf), len(yf))
                    differences.append(auroc(sf[jf], yf[jf]) - auroc(sb[ib], yb[ib]))
            n_paired, paired = paired_difference(cond, base_result, ft_result, sb, yb, sf, yf, paired_rng)
        unpaired = format_difference(delta, differences)
        out.append(f"| {k} | {base_result['auroc']:.3f} | {ft_result['auroc']:.3f} | {unpaired} | "
                   f"{n_paired} | {paired} | {stop_summary(base_result)} | {stop_summary(ft_result)} |")
    out.append("")
with open(sys.argv[3], "w", encoding="utf-8") as stream:
    stream.write("\n".join(out))
print("\n".join(out))

"""실패 탐지기 평가 (Experiment.md 5장, PROPOSAL 4장).

신호 (chunk k를 step t에 추론)
  A: 1 − 가치                          chunk k 추론 직후에 알 수 있음
  B: 예측 관절과 t+16 실제 관절의 거리     chunk k 실행 후 (다음 chunk 결정 때 씀)
  C: 예측 latent와 실제 latent의 MSE      chunk k 실행 후
  F: seed만 바꾼 표본들의 미래 latent 분산  chunk k 추론 직후 (F 데이터가 있을 때)
결정 시점(chunk k를 실행하기 직전)에 알 수 있는 값: A_1..k, F_1..k, B_1..k−1, C_1..k−1.
모든 탐지기는 이 정보만 써서 점수를 낸다. 그래서 실시간으로 돌렸을 때와 같은 결과다.

보정: calib 성공 에피소드에서 (지시문, chunk 순번)별 μ, σ → z-score.
      표본 5개 미만이면 같은 지시문의 k 이하 순번, 그래도 부족하면 같은 k의 모든 지시문을 합친다.
      σ 하한 = calib 성공 전체 σ의 10%.
점수: R_k = 결정 시점 k까지 알 수 있는 z의 최댓값 (비교군 A).
WAC 입력: 결정 시점 k의 [z_A,k, z_F,k, z_B,k−1, z_C,k−1] (z-score).
     calib으로 학습, 지시문 5묶음 교차검증(평가 지시문은 학습에서 뺌), 에피소드마다 무게가 같게.
D-WAC, D-hidden: D에서 hidden feature 또는 WAC 신호를 뺀 비교 (정보별 기여 확인).
hidden: DiT 블록 출력, 마지막 denoising step, 공간·프레임 평균)를 PCA로 줄여 logistic regression.
     chunk k 추론 직후에 알 수 있음. PCA와 표준화는 학습 묶음 calib으로만 맞춘다.
D:   WAC 신호와 hidden PCA 성분을 모두 함께 넣는 logistic regression. 이것이 탐지기다.
임계값: calib 성공 에피소드 최종 점수의 95% 분위수.
조기 중단: 점수가 임계값을 넘은 chunk를 실행하기 전에 멈춘다고 보고, 실패 에피소드에서 아낀 step과
     성공 에피소드를 잘못 멈춘 비율, 중단 후 성공률을 계산한다.

사용: python wac/evaluate.py <데이터 폴더> <정책 통계 json> <출력 json> [--f_dir <F 데이터 폴더>]
"""

import argparse
import glob
import json
import os

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("data_dir")
p.add_argument("stats")
p.add_argument("out")
p.add_argument("--f_dir", default="")
p.add_argument("--boot", type=int, default=2000)
p.add_argument("--pca", type=int, default=32)
p.add_argument("--only_with_f", action="store_true", help="F, hidden이 있는 에피소드만 쓴다 (수집 중 시험용)")
a = p.parse_args()

CHUNK, MIN_N, FLOOR_FRAC, Q = 16, 5, 0.10, 0.95
rng = np.random.default_rng(0)
st = json.load(open(a.stats))
PMIN, PMAX = np.array(st["proprio_min"]), np.array(st["proprio_max"])
norm_p = lambda x: 2 * (np.asarray(x) - PMIN) / (PMAX - PMIN) - 1


# ---------------- 데이터 읽기 ----------------
def load_split(root, cond, split):
    eps = []
    for jl in sorted(glob.glob(os.path.join(root, cond, split, "D*", "episodes.jsonl"))):
        d = os.path.dirname(jl)
        # 중단 후 다시 돌린 작업은 같은 에피소드를 다시 적으므로 마지막 기록만 쓴다
        recs = {}
        for line in open(jl, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                recs[r["episode"]] = r
        for e in recs.values():
            z = np.load(os.path.join(d, e["episode"] + ".npz"))
            t = z["chunk_t"].astype(int)
            K = len(t)
            A = 1 - z["value"].astype(float)
            B = np.full(K, np.nan)
            prop = z["proprio"]
            for k in range(K):
                if t[k] + CHUNK < len(prop):
                    B[k] = np.linalg.norm(z["future_proprio_norm"][k] - norm_p(prop[t[k] + CHUNK]))
            C = np.full(K, np.nan)
            act = os.path.join(d, e["episode"] + "_actual.npz")
            if os.path.exists(act):
                za = np.load(act)
                pos = {int(tt): i for i, tt in enumerate(za["chunk_t"])}
                for k in range(K):
                    if int(t[k]) in pos:
                        C[k] = za["d_img"][pos[int(t[k])]]
            F = np.full(K, np.nan)
            H = None
            if a.f_dir:
                fz = os.path.join(a.f_dir, cond, split, os.path.basename(d), e["episode"] + ".npz")
                if os.path.exists(fz):
                    zf = np.load(fz)
                    if np.array_equal(zf["chunk_t"], z["chunk_t"]):
                        if "extra_latent" in zf.files:
                            # 원래 표본 + 추가 표본의 미래 손목·3인칭 latent 분산 (원소별 분산의 평균)
                            allz = np.concatenate([zf["latent"][:, None, :, 1:3], zf["extra_latent"][:, :, :, 1:3]], 1)
                            F = allz.astype(np.float32).var(axis=1).mean(axis=(1, 2, 3, 4))
                        if "hidden" in zf.files:
                            # (K, 블록, 9 프레임, 2048) → 블록별 프레임 평균을 이어 붙임 (K, 블록×2048)
                            H = zf["hidden"].astype(np.float32).mean(axis=2).reshape(K, -1)
            eps.append({"task": e["task_id"], "fail": int(not e["success"]), "K": K, "t": t,
                        "T_end": len(prop) - 1, "H": H,
                        "sig": {"A": A, "B": B, "C": C, "F": F}, "name": f"{cond}/{split}/{e['episode']}"})
    return eps


# ---------------- 보정 ----------------
def calibrate(calib, sig):
    succ = [e for e in calib if not e["fail"]]
    vals = {}  # (task, k) -> list
    for e in succ:
        for k, v in enumerate(e["sig"][sig]):
            if np.isfinite(v):
                vals.setdefault((e["task"], k), []).append(v)
    allv = np.array([v for vs in vals.values() for v in vs]) if vals else np.array([0.0])
    floor = max(FLOOR_FRAC * allv.std(), 1e-8)
    cache = {}

    def mu_sd(task, k):
        if (task, k) in cache:
            return cache[(task, k)]
        cands = [vals.get((task, k), []),
                 [v for (t2, k2), vs in vals.items() if t2 == task and k2 <= k for v in vs],
                 [v for (t2, k2), vs in vals.items() if k2 == k for v in vs],
                 list(allv)]
        x = next(np.array(c) for c in cands if len(c) >= MIN_N or c is cands[-1])
        r = (x.mean(), max(x.std(), floor))
        cache[(task, k)] = r
        return r

    return mu_sd


def zscores(eps, calib, sigs):
    for s in sigs:
        f = calibrate(calib, s)
        for e in eps:
            e.setdefault("z", {})[s] = np.array([(v - f(e["task"], k)[0]) / f(e["task"], k)[1]
                                                  if np.isfinite(v) else np.nan
                                                  for k, v in enumerate(e["sig"][s])])


# 결정 시점 k의 입력: A_k, F_k, B_{k-1}, C_{k-1}
NOW, LAG = ("A", "F"), ("B", "C")


def decision_inputs(e, sigs):
    K = e["K"]
    cols = []
    for s in sigs:
        z = e["z"][s]
        cols.append(z if s in NOW else np.concatenate([[np.nan], z[:-1]]))
    return np.stack(cols, 1) if cols else np.zeros((K, 0))  # (K, n_sig)


def running_max(x):
    out, m = np.empty(len(x)), -np.inf
    for i, v in enumerate(x):
        if np.isfinite(v):
            m = max(m, v)
        out[i] = m
    return out


# ---------------- logistic regression (numpy IRLS) ----------------
def fit_logreg(X, y, w, l2=1e-2, iters=50):
    n, d = X.shape
    A_ = np.hstack([X, np.ones((n, 1))])
    w = w / w.sum()
    reg = np.full(d + 1, 2 * l2); reg[-1] = 0
    b = np.zeros(d + 1)
    for _ in range(iters):
        pr = 1 / (1 + np.exp(-np.clip(A_ @ b, -30, 30)))
        g = A_.T @ (w * (pr - y)) + reg * b
        H = (A_ * (w * pr * (1 - pr))[:, None]).T @ A_ + np.diag(reg) + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(H, g)
        b -= step
        if np.abs(step).max() < 1e-8:
            break
    return b


def logit_scores(e, b, sigs):
    X = np.nan_to_num(decision_inputs(e, sigs), nan=0.0)   # 아직 없는 값(첫 chunk의 B, C 등)은 평균(0)으로
    return np.hstack([X, np.ones((len(X), 1))]) @ b


# ---------------- 지표 ----------------
def auroc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    d = pos[:, None] - neg[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def auprc(s, y):
    o = np.argsort(-s, kind="stable")
    yy = y[o]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / max(yy.sum(), 1))


def ci(stat_fn, idx_n):
    vals = [stat_fn(rng.integers(0, idx_n, idx_n)) for _ in range(a.boot)]
    vals = np.array([v for v in vals if np.isfinite(v)])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if len(vals) else [np.nan, np.nan]


def evaluate(cond, calib, ev, detectors):
    y = np.array([e["fail"] for e in ev])
    fails = y == 1
    res, finals = {}, {}
    maxK = max(e["K"] for e in ev)
    for name, score_fn in detectors.items():
        cal_final = np.array([score_fn(e)[-1] for e in calib if not e["fail"]])
        cal_final = cal_final[np.isfinite(cal_final)]
        tau = float(np.quantile(cal_final, Q)) if len(cal_final) else np.inf
        R = [score_fn(e) for e in ev]
        final = np.array([r[-1] for r in R])
        finals[name] = final
        det_k = [int(np.argmax(r > tau)) if (r > tau).any() else None for r in R]   # 0부터 센 chunk 순번
        # 같은 chunk 순번까지 살아 있는 에피소드끼리 비교한 AUROC (에피소드 길이 효과 제거)
        alive = []
        for k in range(1, min(maxK, 14) + 1):
            idx = [i for i, e in enumerate(ev) if e["K"] >= k]
            sk = np.nan_to_num(np.array([R[i][k - 1] for i in idx]), neginf=-1e9)
            alive.append({"k": k, "n": len(idx), "n_fail": int(y[idx].sum()), "auroc": auroc(sk, y[idx])})
        # 조기 중단: 경고한 chunk를 실행하기 전에 멈춘다
        stop_t = [int(ev[i]["t"][d]) if d is not None else None for i, d in enumerate(det_k)]
        stopped = np.array([s is not None for s in stop_t])
        saved = np.array([(ev[i]["T_end"] - stop_t[i]) if (fails[i] and stopped[i]) else 0
                          for i in range(len(ev))], dtype=float)
        fail_steps = np.array([ev[i]["T_end"] for i in range(len(ev)) if fails[i]], dtype=float)
        f = np.nan_to_num(final, neginf=-1e9)
        det_steps = [stop_t[i] for i in range(len(ev)) if fails[i] and stopped[i]]
        res[name] = {
            "auroc": auroc(f, y), "auroc_ci": ci(lambda i: auroc(f[i], y[i]), len(y)),
            "auprc": auprc(f, y), "auprc_ci": ci(lambda i: auprc(f[i], y[i]), len(y)),
            "threshold": tau,
            "false_alarm": float(stopped[~fails].mean()),
            "detection_rate": float(stopped[fails].mean()),
            "median_detect_step": float(np.median(det_steps)) if det_steps else None,
            "saved_steps_per_failure": float(saved[fails].mean()),
            "saved_fraction_of_failure_steps": float(saved[fails].sum() / fail_steps.sum()),
            "success_rate_before": float((~fails).mean()),
            "success_rate_after_stop": float(((~fails) & ~stopped).mean()),
            "alive_auroc": alive,
        }
        # 시점별 임계값 (SAFE, FAIL-Detect의 time-varying conformal band): chunk 순번 k마다
        # calib 성공 에피소드의 R_k 95% 분위수. 이미 끝난 에피소드는 마지막 점수를 유지한다.
        cal_R = [score_fn(e) for e in calib if not e["fail"]]
        # 띠 전체로 calib 성공 에피소드의 95%가 한 번도 넘지 않도록, chunk별 분위수 수준 q를 올려 가며 고른다
        def band(q):
            return np.array([np.quantile([r[min(k, len(r) - 1)] for r in cal_R], q) for k in range(maxK)])

        def crossing(tk):
            return np.mean([any(r[k] > tk[min(k, maxK - 1)] for k in range(len(r))) for r in cal_R])

        q_band = next((q for q in np.arange(Q, 1.0001, 0.0025) if crossing(band(min(q, 1.0))) <= 1 - Q), 1.0)
        tau_k = band(min(q_band, 1.0))
        det_tv = [next((k for k in range(len(r)) if r[k] > tau_k[min(k, maxK - 1)]), None) for r in R]
        st_tv = [int(ev[i]["t"][d]) if d is not None else None for i, d in enumerate(det_tv)]
        stp = np.array([s is not None for s in st_tv])
        sv = np.array([(ev[i]["T_end"] - st_tv[i]) if (fails[i] and stp[i]) else 0 for i in range(len(ev))], float)
        dsteps = [st_tv[i] for i in range(len(ev)) if fails[i] and stp[i]]
        res[name]["time_varying"] = {
            "false_alarm": float(stp[~fails].mean()), "detection_rate": float(stp[fails].mean()),
            "median_detect_step": float(np.median(dsteps)) if dsteps else None,
            "saved_fraction_of_failure_steps": float(sv[fails].sum() / fail_steps.sum()),
            "success_rate_after_stop": float(((~fails) & ~stp).mean()),
            "band_quantile": float(min(q_band, 1.0)),
        }
    for name in detectors:
        if name == "A":
            continue
        fa, fb = np.nan_to_num(finals["A"], neginf=-1e9), np.nan_to_num(finals[name], neginf=-1e9)
        res[name]["auroc_minus_A"] = auroc(fb, y) - auroc(fa, y)
        res[name]["auroc_minus_A_ci"] = ci(lambda i: auroc(fb[i], y[i]) - auroc(fa[i], y[i]), len(y))
    return res


# ---------------- hidden feature ----------------
def pca_fit(X, n_max):
    """표준화 후 PCA. 반환 함수 proj(Z, n)은 앞 n개 성분을 준다."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    _, _, vt = np.linalg.svd((X - mu) / sd, full_matrices=False)
    vt = vt[:n_max]
    return lambda Z, n: ((Z - mu) / sd) @ vt[:n].T


PCA_GRID = [4, 8, 16, 32]
L2_GRID = [1e-2, 1e-1, 1.0, 10.0]


# ---------------- 실행 ----------------
out = {}
for cond in ["obj_0.090", "wristroll_50"]:
    calib, ev = load_split(a.data_dir, cond, "calib"), load_split(a.data_dir, cond, "eval")
    if a.only_with_f:
        calib = [e for e in calib if e["H"] is not None]
        ev = [e for e in ev if e["H"] is not None]
        if not calib or not ev:
            print(f"{cond}: F 데이터 없음, 건너뜀")
            continue
    has_f = all(np.isfinite(e["sig"]["F"]).any() for e in calib + ev)
    has_h = all(e["H"] is not None for e in calib + ev)
    sigs = ["A", "B", "C"] + (["F"] if has_f else [])
    zscores(calib + ev, calib, sigs)  # 보정은 calib 성공 에피소드로만

    def single(s):
        return lambda e: running_max(decision_inputs(e, [s])[:, 0])

    dets = {"A": single("A")}   # 비교군: 모델 자체 가치 예측

    # 학습하는 탐지기: 지시문 5묶음 교차검증. 평가 지시문을 뺀 calib으로 학습한다.
    tasks = sorted({e["task"] for e in calib})
    folds = [tasks[i::5] for i in range(5)]
    # D: 모든 정보(WAC 신호 + hidden feature)를 함께. D-WAC, D-hidden은 한쪽을 뺀 비교
    kinds = ["D", "D-WAC", "D-hidden"] if has_h else ["D-WAC"]
    models = {k: {} for k in kinds}

    def feats(e, kind, proj, n):
        parts = []
        if kind in ("D", "D-WAC"):
            parts.append(np.nan_to_num(decision_inputs(e, sigs), nan=0.0))
        if kind in ("D", "D-hidden"):
            parts.append(proj(e["H"], n))
        return np.hstack(parts)

    def fit(tr, kind, proj, n, l2):
        X = np.vstack([feats(e, kind, proj, n) for e in tr])
        yy = np.concatenate([[e["fail"]] * e["K"] for e in tr])
        ww = np.concatenate([[1 / e["K"]] * e["K"] for e in tr])
        return fit_logreg(X, yy, ww, l2=l2)

    def final_score(e, kind, b, proj, n):
        X = feats(e, kind, proj, n)
        return np.max(np.hstack([X, np.ones((len(X), 1))]) @ b)

    def select(tr, kind):
        """PCA 차원과 L2 세기를 학습 묶음 안의 지시문 단위 내부 교차검증 AUROC로 고른다 (eval은 쓰지 않는다)."""
        tt = sorted({e["task"] for e in tr})
        inner = [tt[i::4] for i in range(4)]
        grid = [(n, l2) for n in (PCA_GRID if kind != "D-WAC" else [0]) for l2 in L2_GRID]
        scores = {g: ([], []) for g in grid}
        for ifold in inner:
            itr = [e for e in tr if e["task"] not in ifold]
            iva = [e for e in tr if e["task"] in ifold]
            proj = pca_fit(np.vstack([e["H"] for e in itr]), max(PCA_GRID)) if kind != "D-WAC" else None
            for g in grid:
                b = fit(itr, kind, proj, *g)
                scores[g][0].extend(final_score(e, kind, b, proj, g[0]) for e in iva)
                scores[g][1].extend(e["fail"] for e in iva)
        return max(grid, key=lambda g: auroc(np.array(scores[g][0]), np.array(scores[g][1])))

    chosen = {k: [] for k in kinds}
    for fold in folds:
        tr = [e for e in calib if e["task"] not in fold]
        proj = pca_fit(np.vstack([e["H"] for e in tr]), max(PCA_GRID)) if has_h else None
        for kind in kinds:
            n, l2 = select(tr, kind)
            chosen[kind].append((n, l2))
            b = fit(tr, kind, proj, n, l2)
            for t_ in fold:
                models[kind][t_] = (b, proj, n)

    def learned(kind):
        def f(e):
            b, proj, n = models[kind][e["task"]]
            X = feats(e, kind, proj, n)
            return running_max(np.hstack([X, np.ones((len(X), 1))]) @ b)
        return f

    for kind in kinds:
        dets[kind] = learned(kind)

    res = evaluate(cond, calib, ev, dets)
    res["_meta"] = {"signals": sigs, "hidden": has_h, "chosen_pca_l2": {k: [list(map(float, c)) for c in v] for k, v in chosen.items()}, "n_calib": len(calib), "n_eval": len(ev),
                    "eval_fail": int(sum(e["fail"] for e in ev)),
                    "D_coef_mean": dict(zip(sigs + ["bias"],
                                            np.mean([models["D-WAC"][t][0] for t in tasks], 0).round(3).tolist()))}
    out[cond] = res
    print(f"\n== {cond}: calib {len(calib)}, eval {len(ev)} (실패 {res['_meta']['eval_fail']}), 신호 {sigs}, hidden {has_h}")
    print("탐지기  AUROC  95% CI         오경보 탐지율 탐지step 아낀동작 중단후성공  A 대비 AUROC 차")
    for n, r in res.items():
        if n.startswith("_"):
            continue
        dA = (f"{r['auroc_minus_A']:+.3f} [{r['auroc_minus_A_ci'][0]:+.3f}, {r['auroc_minus_A_ci'][1]:+.3f}]"
              if "auroc_minus_A" in r else "")
        print(f"{n:<6} {r['auroc']:.3f} [{r['auroc_ci'][0]:.3f},{r['auroc_ci'][1]:.3f}] {r['false_alarm']:>5.2f} "
              f"{r['detection_rate']:>5.2f} {str(r['median_detect_step']):>7} {r['saved_fraction_of_failure_steps']:>7.2f} "
              f"   {r['success_rate_before']:.2f}->{r['success_rate_after_stop']:.2f}  {dA}")
    print("시점별 임계값: 오경보 / 탐지율 / 탐지step / 아낀동작 / 중단후성공")
    for n, r in res.items():
        if not n.startswith("_"):
            tv = r["time_varying"]
            print(f"  {n:<6} {tv['false_alarm']:.2f} {tv['detection_rate']:.2f} {tv['median_detect_step']} "
                  f"{tv['saved_fraction_of_failure_steps']:.2f} {tv['success_rate_after_stop']:.2f}")
    print("살아 있는 에피소드끼리 AUROC (k=2,4,6,8):",
          {n: [round(x["auroc"], 3) for x in r["alive_auroc"] if x["k"] in (2, 4, 6, 8)]
           for n, r in res.items() if not n.startswith("_")})
    print("D 계수(평균):", res["_meta"]["D_coef_mean"])
    print("고른 (PCA 차원, L2):", res["_meta"]["chosen_pca_l2"])

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False, default=float)
print("\n저장:", a.out)

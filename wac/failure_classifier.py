"""실패 탐지 분류기 (PROPOSAL 5.3의 조건 D, E).

입력은 chunk 단위 표다. 한 행이 (에피소드, chunk) 하나다.
    X        (N, D)  chunk k 시점까지 알 수 있는 feature
                     D 조건: [z_A, z_B, z_C]  (보정으로 만든 z-score)
                     E 조건: DiT 중간층 feature를 프레임별로 평균 낸 벡터
    episode  (N,)    에피소드 번호
    task     (N,)    과제 번호 (교차검증 묶음에 씀)
    chunk    (N,)    에피소드 안의 chunk 순번
    y        (N,)    에피소드가 실패면 1, 성공이면 0 (같은 에피소드의 chunk는 전부 같은 값)

만드는 것:
    1. 표준화 (+ E는 PCA로 차원 축소)
    2. L2 정규화 logistic regression. 에피소드마다 무게가 같도록 chunk에 1/(chunk 수) 가중치
    3. chunk 확률 p(k)의 누적 최댓값 R(k) = max_{j<=k} p(j) 를 에피소드 점수로 사용
    4. 과제 단위 교차검증: 평가 과제는 학습에 절대 안 들어감. 정규화 세기도 학습 과제 안에서 고름

scikit-learn 없이 PyTorch만 쓴다 (공용 venv에 패키지를 늘리지 않기 위해).
"""

import numpy as np
import torch


# ---------- 전처리 ----------

class Preprocess:
    """학습 데이터로만 평균, 표준편차, PCA 축을 정하고 평가 데이터에는 그대로 적용한다."""

    def __init__(self, n_pca=None):
        self.n_pca = n_pca

    def fit(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        Z = (X - self.mu) / self.sd
        if self.n_pca and self.n_pca < X.shape[1]:
            # SVD로 주성분 축을 구한다. 축은 학습 데이터에서만 정한다.
            _, _, vt = np.linalg.svd(Z, full_matrices=False)
            self.axes = vt[: self.n_pca].T
        else:
            self.axes = None
        return self

    def transform(self, X):
        Z = (X - self.mu) / self.sd
        return Z @ self.axes if self.axes is not None else Z


# ---------- logistic regression ----------

def fit_logreg(X, y, w, l2):
    """가중 logistic regression. 손실 = sum_i w_i * BCE(sigmoid(x_i·θ + b), y_i) / sum w + l2 * |θ|²"""
    X = torch.tensor(X, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    w = torch.tensor(w, dtype=torch.float64)
    w = w / w.sum()
    theta = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([theta, b], lr=1.0, max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = X @ theta + b
        loss = (w * torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")).sum()
        loss = loss + l2 * (theta ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return theta.detach().numpy(), float(b.detach())


def predict_logreg(X, theta, b):
    return 1.0 / (1.0 + np.exp(-(X @ theta + b)))


# ---------- 에피소드 점수와 지표 ----------

def episode_weights(episode):
    """에피소드마다 무게 합이 1이 되게: 실패 에피소드가 chunk가 많아 표를 독차지하는 것을 막는다."""
    _, inv, counts = np.unique(episode, return_inverse=True, return_counts=True)
    return 1.0 / counts[inv]


def running_max(p, episode, chunk):
    """chunk 확률을 에피소드 안에서 chunk 순서대로 누적 최댓값으로 바꾼다. R(k) = max_{j<=k} p(j)"""
    R = np.empty_like(p)
    for e in np.unique(episode):
        idx = np.where(episode == e)[0]
        idx = idx[np.argsort(chunk[idx])]
        R[idx] = np.maximum.accumulate(p[idx])
    return R


def final_scores(R, episode, chunk, y):
    """에피소드별 최종 점수(마지막 chunk의 R)와 라벨."""
    eps = np.unique(episode)
    s = np.array([R[episode == e][np.argmax(chunk[episode == e])] for e in eps])
    lab = np.array([y[episode == e][0] for e in eps])
    return eps, s, lab


def auroc(score, label):
    """Mann-Whitney 방식 AUROC. 실패(1)의 점수가 성공(0)보다 클 확률. 동점은 0.5."""
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


# ---------- 과제 단위 교차검증 ----------

def task_folds(task, n_folds):
    tasks = np.unique(task)
    rng = np.random.default_rng(0)
    rng.shuffle(tasks)
    return [set(t) for t in np.array_split(tasks, n_folds)]


def train_eval_fold(X, y, episode, task, chunk, train_mask, l2, n_pca):
    pre = Preprocess(n_pca).fit(X[train_mask])
    Xtr = pre.transform(X[train_mask])
    theta, b = fit_logreg(Xtr, y[train_mask], episode_weights(episode[train_mask]), l2)
    return pre, theta, b


def choose_l2(X, y, episode, task, chunk, n_pca, l2_grid, n_inner=3):
    """학습 과제만 써서 정규화 세기를 고른다 (안쪽 교차검증). 평가 과제는 여기서 쓰지 않는다."""
    best, best_auc = l2_grid[0], -1
    inner = task_folds(task, min(n_inner, len(np.unique(task))))
    for l2 in l2_grid:
        aucs = []
        for val_tasks in inner:
            va = np.isin(task, list(val_tasks))
            pre, th, b = train_eval_fold(X, y, episode, task, chunk, ~va, l2, n_pca)
            p = predict_logreg(pre.transform(X[va]), th, b)
            R = running_max(p, episode[va], chunk[va])
            _, s, lab = final_scores(R, episode[va], chunk[va], y[va])
            aucs.append(auroc(s, lab))
        m = np.nanmean(aucs)
        if m > best_auc:
            best, best_auc = l2, m
    return best


def cross_validate(X, y, episode, task, chunk, n_folds=5, n_pca=None,
                   l2_grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
    """바깥 교차검증: 평가 과제 묶음마다 나머지 과제로 학습. 모든 chunk에 대해 '학습에 안 쓰인 모델'의 R(k)를 돌려준다."""
    R_all = np.full(len(y), np.nan)
    info = []
    for test_tasks in task_folds(task, n_folds):
        te = np.isin(task, list(test_tasks))
        tr = ~te
        l2 = choose_l2(X[tr], y[tr], episode[tr], task[tr], chunk[tr], n_pca, l2_grid)
        pre, th, b = train_eval_fold(X, y, episode, task, chunk, tr, l2, n_pca)
        p = predict_logreg(pre.transform(X[te]), th, b)
        R_all[te] = running_max(p, episode[te], chunk[te])
        info.append({"test_tasks": sorted(int(t) for t in test_tasks), "l2": l2,
                     "theta": th if n_pca is None else None})
    _, s, lab = final_scores(R_all, episode, chunk, y)
    return R_all, auroc(s, lab), info


# ---------- 합성 데이터로 동작 확인 ----------

def _synthetic(n_tasks=10, eps_per_task=12, seed=0):
    """실패 에피소드는 어느 chunk부터 z_B, z_C가 커지고, z_A는 약하게만 반응하는 가짜 데이터."""
    rng = np.random.default_rng(seed)
    rows = []
    e = 0
    for t in range(n_tasks):
        for _ in range(eps_per_task):
            fail = rng.random() < 0.5
            n_chunk = 14 if fail else rng.integers(5, 10)
            onset = rng.integers(2, 8)
            for k in range(n_chunk):
                bad = fail and k >= onset
                z = rng.normal(size=3) + (np.array([0.3, 1.5, 1.0]) if bad else 0)
                rows.append((e, t, k, int(fail), *z))
            e += 1
    a = np.array(rows)
    return a[:, 4:], a[:, 3], a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2].astype(int)


if __name__ == "__main__":
    X, y, episode, task, chunk = _synthetic()
    R, auc, info = cross_validate(X, y, episode, task, chunk)
    print(f"[합성 데이터] chunk {len(y)}개, 에피소드 {len(np.unique(episode))}개")
    print(f"과제 단위 교차검증 AUROC = {auc:.3f}")
    for f in info:
        print(f"  평가 과제 {f['test_tasks']}  l2={f['l2']:g}  가중치(z_A, z_B, z_C)={np.round(f['theta'], 2)}")

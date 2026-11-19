"""파인튜닝 체크포인트를 epoch 단위 폴더(E1, E2, ...)로 정리한다.

공식 학습 코드는 <raw>/.../checkpoints/iter_000000600 처럼 step 번호로 저장한다.
save_iter를 1 epoch의 step 수로 맞춰 두면 저장 시점이 곧 epoch 경계이므로,
iter 번호를 epoch 번호로 바꿔 <run>/E1, <run>/E2 ... 에 심볼릭 링크와 설명 파일을 만든다.

사용:
  python organize_epochs.py <run 폴더> <1 epoch step 수>            한 번 정리
  python organize_epochs.py <run 폴더> <1 epoch step 수> --watch    학습 중 계속 감시 (새 체크포인트가 생기면 정리)
"""

import glob
import json
import os
import re
import sys
import time


def organize(run, steps_per_epoch):
    found = []
    for ck in sorted(glob.glob(os.path.join(run, "raw", "**", "checkpoints", "iter_*"), recursive=True)):
        m = re.search(r"iter_(\d+)", os.path.basename(ck))
        if not m:
            continue
        it = int(m.group(1))
        if it == 0 or it % steps_per_epoch:
            continue  # epoch 경계가 아닌 저장은 건너뜀
        ep = it // steps_per_epoch
        dst = os.path.join(run, f"E{ep}")
        os.makedirs(dst, exist_ok=True)
        link = os.path.join(dst, "checkpoint")
        if not os.path.lexists(link):
            os.symlink(os.path.relpath(ck, dst), link)
            info = {"epoch": ep, "iteration": it, "steps_per_epoch": steps_per_epoch,
                    "checkpoint": os.path.relpath(ck, run), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            with open(os.path.join(dst, "info.json"), "w") as f:
                json.dump(info, f, indent=1)
            with open(os.path.join(dst, "README.md"), "w", encoding="utf-8") as f:
                f.write(f"# E{ep}\n\n{ep} epoch 학습 직후 체크포인트 (step {it}).\n\n"
                        f"- 체크포인트: `checkpoint` (원본 `{info['checkpoint']}`)\n"
                        f"- 저장 시각: {info['saved_at']}\n"
                        f"- 검증 손실은 학습 로그와 `../검증.md`에 정리한다.\n")
            print(f"[E{ep}] step {it} -> {dst}", flush=True)
        found.append(ep)
    return found


if __name__ == "__main__":
    run, spe = sys.argv[1], int(sys.argv[2])
    if "--watch" in sys.argv:
        while True:
            organize(run, spe)
            if os.path.exists(os.path.join(run, "TRAINING_DONE")):
                organize(run, spe)
                break
            time.sleep(60)
    else:
        print("정리된 epoch:", organize(run, spe))

"""evaluate.py 결과 json의 에피소드 목록을 수집 폴더(지시문 폴더)와 짝짓는다.

결과 json의 에피소드 이름은 '<조건>/<묶음>/init25_fail'처럼 지시문 폴더가 빠져 있어, 지시문 10개 사이에서 겹친다.
evaluate.py의 load_split은 지시문 폴더를 정렬한 순서로, 폴더 안에서는 episodes.jsonl의 처음 나온 순서로
에피소드를 쌓는다(같은 에피소드를 다시 적으면 마지막 기록으로 덮지만 순서는 처음 위치). 같은 순서를 다시 만들면
json의 i번째 에피소드가 어느 지시문 폴더의 것인지 알 수 있다.
"""

import glob
import json
import os


def ordered_episodes(data_dir, cond, split):
    """[(지시문 폴더 경로, 에피소드 이름, 성공 여부)]를 evaluate.py와 같은 순서로."""
    out = []
    for jl in sorted(glob.glob(os.path.join(data_dir, cond, split, "D*", "episodes.jsonl"))):
        d = os.path.dirname(jl)
        recs = {}
        with open(jl, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    recs[r["episode"]] = r
        for e in recs.values():
            out.append((d, e["episode"], bool(e["success"])))
    return out


def task_dirs_for(result_eps, data_dir, cond, split="eval"):
    """결과 json의 episodes 목록과 같은 길이의 지시문 폴더 목록. 이름과 성공 여부가 하나라도 안 맞으면 오류."""
    ref = ordered_episodes(data_dir, cond, split)
    if len(ref) != len(result_eps):
        raise ValueError(f"{cond}/{split}: 에피소드 수가 다름 ({len(ref)} vs {len(result_eps)})")
    dirs = []
    for (d, ep, succ), r in zip(ref, result_eps):
        if not r["name"].endswith("/" + ep) or bool(r["fail"]) == succ:
            raise ValueError(f"순서 불일치: {r['name']} vs {d}/{ep}")
        dirs.append(d)
    return dirs

import torch

from diffuse_worker.inference import SliceRunner
from diffuse_worker.slicing import ModelSlice

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _load_full_runner():
    s = ModelSlice()
    info = s.load(MODEL, 0, 0, None)
    total = info.total_layers
    s.load(MODEL, 0, total, None)
    return SliceRunner(s), total


def _greedy_no_cache(runner, ids, steps):
    seq = list(ids)
    for _ in range(steps):
        t = torch.tensor([seq], dtype=torch.long)
        logits = runner.run(t, is_input_ids=True, use_cache=False)
        nxt = int(torch.argmax(logits[0, -1]))
        seq.append(nxt)
    return seq


def _greedy_with_cache(runner, ids, steps, session):
    seq = list(ids)
    first = torch.tensor([seq], dtype=torch.long)
    logits = runner.run(first, is_input_ids=True, session_id=session, use_cache=True)
    nxt = int(torch.argmax(logits[0, -1]))
    seq.append(nxt)
    for _ in range(steps - 1):
        last = torch.tensor([[seq[-1]]], dtype=torch.long)
        logits = runner.run(last, is_input_ids=True, session_id=session, use_cache=True)
        nxt = int(torch.argmax(logits[0, -1]))
        seq.append(nxt)
    return seq


def test_cache_matches_no_cache():
    runner_a, _ = _load_full_runner()
    runner_b, _ = _load_full_runner()

    prompt = [9707, 11, 1879, 0]
    steps = 8

    no_cache = _greedy_no_cache(runner_a, prompt, steps)
    with_cache = _greedy_with_cache(runner_b, prompt, steps, "sess1")

    assert no_cache == with_cache, (
        f"cache must match no-cache.\nno_cache={no_cache}\nwith_cache={with_cache}"
    )
"""Compose the night-3 phase-1 heal band (fresh compact-lane flagship run).

Sources, in dedup priority order (first occurrence wins):
  1. retrieval_4k train split  (new dataset; upsampled x3 with distinct ids)
  2. gdnx_heal_1536_cont       (continuation band; partially overlaps xxl)
  3. csa_heal_4096_xxl

The model is a fresh warm-start, so prior-run exposure does not constrain
composition; dedup only removes byte-identical rows so no window is seen
twice per pass. Output: single-shard band + example-id list + manifest.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import torch

DATA = Path.home() / "attn_data" / "data"
OUT = DATA / "gdnx_night3_band"
SEQ_LEN = 4096
RETRIEVAL_REPEATS = 3
SEED = 20260718


def load_band(path: Path, pattern: str = "shard_main_*.pt") -> torch.Tensor:
    shards = sorted(path.glob(pattern))
    if shards:
        return torch.cat([torch.load(p, map_location="cpu", weights_only=True) for p in shards])
    raise FileNotFoundError(path)


def main() -> None:
    retrieval = torch.load(
        DATA / "retrieval" / "blocks_4k" / "train_retrieval_4k.pt",
        map_location="cpu", weights_only=True,
    ).to(torch.int32)
    cont = load_band(DATA / "gdnx_heal_1536_cont").to(torch.int32)
    xxl = load_band(DATA / "csa_heal_4096_xxl").to(torch.int32)
    for name, band in (("retrieval", retrieval), ("cont", cont), ("xxl", xxl)):
        assert band.shape[1] == SEQ_LEN, f"{name}: {tuple(band.shape)}"

    seen: set[str] = set()
    rows: list[torch.Tensor] = []
    ids: list[str] = []
    dropped = 0
    for source, band in (("ret", retrieval), ("cont", cont), ("xxl", xxl)):
        for i in range(band.shape[0]):
            row = band[i]
            digest = hashlib.sha1(row.numpy().tobytes()).hexdigest()
            if digest in seen:
                dropped += 1
                continue
            seen.add(digest)
            rows.append(row)
            ids.append(f"n3-{source}-{i:05d}")

    # Upsample the retrieval rows: they are already first in `rows`; append
    # extra passes with distinct ids so window accounting stays unique.
    n_ret = int(sum(1 for x in ids if x.startswith("n3-ret-")))
    for repeat in range(1, RETRIEVAL_REPEATS):
        for i in range(n_ret):
            rows.append(rows[i])
            ids.append(f"{ids[i]}-r{repeat}")

    order = list(range(len(rows)))
    random.Random(SEED).shuffle(order)
    stacked = torch.stack([rows[i] for i in order])
    shuffled_ids = [ids[i] for i in order]

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(stacked, OUT / "shard_main_0000.pt")
    (OUT / "example_ids.json").write_text(json.dumps(shuffled_ids))
    manifest = {
        "band": "gdnx_night3_phase1",
        "seq_len": SEQ_LEN,
        "seed": SEED,
        "rows": stacked.shape[0],
        "unique_rows": len(seen),
        "duplicates_dropped": dropped,
        "retrieval_rows": n_ret,
        "retrieval_repeats": RETRIEVAL_REPEATS,
        "sources": ["retrieval/blocks_4k(train)", "gdnx_heal_1536_cont", "csa_heal_4096_xxl"],
        "sha256_rows": hashlib.sha256(stacked.numpy().tobytes()).hexdigest(),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()

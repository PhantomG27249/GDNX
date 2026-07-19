#!/usr/bin/env python3
"""Post periodic GDN-X training metrics to a Discord webhook.

Runs as a separate process so a webhook/network failure can never touch the
training run. The trainer emits one JSON line per update into
OUTPUT/live/<job_id>.jsonl when GDNX_LIVE_METRICS=1; this script tails every
such file and posts a summary embed either every --every-updates updates or
every --every-minutes minutes, whichever comes first, plus a final message
when a job reaches its update budget.

The webhook URL is read from a file named 'webhook' in the repo root (or the
path given via --webhook-file). Stdlib only.

Usage:
    python metrics_webhook.py --output /path/to/run/output [--every-updates 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LOSS_KEYS = ("total", "ce", "kl", "layerwise")


def read_webhook_url(path: Path) -> str:
    url = path.read_text(encoding="utf-8").strip()
    if not url.startswith("https://"):
        raise ValueError(f"webhook file {path} does not contain an https URL")
    return url


def post_discord(url: str, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "gdnx-metrics/1"},
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                response.read()
            return
        except urllib.error.HTTPError as error:
            if error.code == 429:
                retry_after = 2.0
                try:
                    retry_after = float(json.loads(error.read()).get("retry_after", 2.0))
                except Exception:
                    pass
                time.sleep(retry_after + 0.5)
                continue
            print(f"[metrics_webhook] discord HTTP {error.code}", file=sys.stderr)
            return
        except Exception as error:
            print(f"[metrics_webhook] post failed: {error}", file=sys.stderr)
            time.sleep(2.0 * (attempt + 1))
    print("[metrics_webhook] giving up on this post", file=sys.stderr)


class JobTail:
    """Incremental reader for one job's live JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.records: list[dict] = []
        self.last_posted_update = 0
        self.last_post_time = 0.0
        self.finished_announced = False

    def poll(self) -> list[dict]:
        try:
            size = self.path.stat().st_size
            if size < self.offset:  # truncated/restarted job: start over
                self.offset = 0
                self.records.clear()
            if size == self.offset:
                return []
            with self.path.open("r", encoding="utf-8") as handle:
                handle.seek(self.offset)
                chunk = handle.read()
                # Only consume complete lines; keep a partial tail for later.
                complete, _, partial = chunk.rpartition("\n")
                self.offset = size - len(partial.encode("utf-8"))
        except OSError:
            return []
        fresh = []
        for line in complete.splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and "update" in record:
                fresh.append(record)
        self.records.extend(fresh)
        return fresh


def format_embed(tail: JobTail, latest: dict) -> dict:
    losses = latest.get("losses", {})
    fields = [
        {
            "name": "progress",
            "value": f"update {latest['update']}/{latest.get('max_updates', '?')} · "
                     f"{latest.get('tokens_seen', 0):,} tokens",
            "inline": False,
        },
        {
            "name": "losses",
            "value": " · ".join(
                f"{key} {losses[key]:.4f}" for key in LOSS_KEYS if key in losses
            ) or "n/a",
            "inline": False,
        },
    ]
    # Throughput + ETA from the window since the last post.
    window = [r for r in tail.records if r["update"] > tail.last_posted_update]
    if len(window) >= 2 and window[-1].get("wall_time") and window[0].get("wall_time"):
        seconds = window[-1]["wall_time"] - window[0]["wall_time"]
        tokens = window[-1].get("tokens_seen", 0) - window[0].get("tokens_seen", 0)
        updates = window[-1]["update"] - window[0]["update"]
        if seconds > 0 and updates > 0:
            per_update = seconds / updates
            remaining = latest.get("max_updates", latest["update"]) - latest["update"]
            eta_h = remaining * per_update / 3600
            fields.append({
                "name": "pace",
                "value": f"{tokens / seconds:,.0f} tok/s · {per_update:.0f} s/update · "
                         f"ETA {eta_h:.1f} h",
                "inline": False,
            })
    if latest.get("skipped_steps"):
        fields.append({
            "name": "warnings",
            "value": f"skipped_steps={latest['skipped_steps']}",
            "inline": False,
        })
    arm = latest.get("arm", "?")
    return {
        "title": f"GDN-X · {latest.get('job_id', tail.path.stem)}",
        "description": f"arm `{arm}`",
        "fields": fields,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True,
                        help="training output directory (contains live/)")
    parser.add_argument("--webhook-file", type=Path, default=REPO_ROOT / "webhook")
    parser.add_argument("--every-updates", type=int, default=8,
                        help="post after this many new updates per job")
    parser.add_argument("--every-minutes", type=float, default=20.0,
                        help="also post if this many minutes elapse with new data")
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--once", action="store_true",
                        help="single poll+post pass (for testing)")
    args = parser.parse_args()

    url = read_webhook_url(args.webhook_file)
    live_dir = args.output / "live"
    tails: dict[Path, JobTail] = {}
    print(f"[metrics_webhook] watching {live_dir}", file=sys.stderr)

    while True:
        try:
            paths = sorted(live_dir.glob("*.jsonl")) if live_dir.is_dir() else []
            for path in paths:
                tail = tails.setdefault(path, JobTail(path))
                tail.poll()
                if not tail.records:
                    continue
                latest = tail.records[-1]
                new_updates = latest["update"] - tail.last_posted_update
                stale = time.time() - tail.last_post_time > args.every_minutes * 60
                finished = latest["update"] >= latest.get("max_updates", float("inf"))
                if finished and not tail.finished_announced:
                    embed = format_embed(tail, latest)
                    embed["title"] = "✅ " + embed["title"] + " — complete"
                    post_discord(url, {"embeds": [embed]})
                    tail.finished_announced = True
                    tail.last_posted_update = latest["update"]
                    tail.last_post_time = time.time()
                elif new_updates >= args.every_updates or (new_updates > 0 and stale):
                    post_discord(url, {"embeds": [format_embed(tail, latest)]})
                    tail.last_posted_update = latest["update"]
                    tail.last_post_time = time.time()
        except Exception as error:
            print(f"[metrics_webhook] loop error: {error}", file=sys.stderr)
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

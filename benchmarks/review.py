"""Attach index/criteria-bound semantic reviews without rerunning paid model work."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from .quality import decide_quality
from .run import summarize


def apply_reviews(report, reviews):
    result = deepcopy(report)
    by_id = {}
    for review in reviews:
        key = (review["index_id"], review["criteria_hash"])
        if key in by_id:
            raise ValueError("duplicate semantic review identity")
        by_id[key] = review
    used = set()
    for row in result["runs"]:
        quality = row.get("quality")
        if quality is None:
            continue
        key = (quality["index_id"], quality["criteria_hash"])
        if key in by_id:
            decide_quality(quality, by_id[key])
            used.add(key)
    if set(by_id) != used:
        raise ValueError("review references an index or criterion absent from this report")
    result["summary"] = summarize(result["runs"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "reviews",
        type=Path,
        help="JSON list with index_id, criteria_hash, reviewer, verdict approved/rejected",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    reviews = json.loads(args.reviews.read_text(encoding="utf-8"))
    output = apply_reviews(report, reviews)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

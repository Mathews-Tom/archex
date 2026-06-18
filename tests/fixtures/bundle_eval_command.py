from __future__ import annotations

import json
import sys
import time


def main() -> int:
    mode = sys.argv[1]
    if mode == "invalid-json":
        sys.stdout.write("not json")
        return 0
    if mode == "bad-schema":
        sys.stdout.write(json.dumps({"answer": "missing fields"}))
        return 0
    if mode == "sleep":
        time.sleep(5)
        return 0
    payload = json.loads(sys.stdin.read())
    if mode == "exit-2":
        return 2
    answer = "bundle contains receipt"
    success = "pass" if payload["question"] and payload["receipt_json"] != "null" else "fail"
    output = {
        "answer": answer,
        "confidence": 0.95,
        "needed_files": ["src/frontier.py"],
        "attempted_more_context": False,
        "post_bundle_read_turns": 1,
    }
    if mode != "no-success":
        output["bundle_only_success"] = success
    sys.stdout.write(json.dumps(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

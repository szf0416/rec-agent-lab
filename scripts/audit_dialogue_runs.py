"""从保存的冻结任务结果复现失败与回退贡献审计。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recagent.eval.audit import audit_runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="冻结对话结果配对审计")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--imported", type=Path)
    parser.add_argument("--default", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    paths = {"baseline": args.baseline, "imported": args.imported, "default": args.default}
    runs = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
        if path is not None
    }
    payload = json.dumps(audit_runs(runs), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

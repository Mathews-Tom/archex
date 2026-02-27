from __future__ import annotations

import sys

from packages.core.src.index import initialize


def main(args: list[str] | None = None) -> int:
    argv = args if args is not None else sys.argv[1:]
    debug = "--debug" in argv
    config = initialize("cli-app", debug=debug)
    print(f"Starting CLI with config: {config.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Placeholder script retained for legacy workflows.

MusicNN-based tagging has been removed from the comprehensive analyzer, so this
utility simply informs the caller instead of exercising the old dependency
stack.
"""


def main() -> int:
    print("MusicNN support has been removed from the comprehensive analyzer.")
    print("No further action is required; this script is kept for backwards compatibility.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
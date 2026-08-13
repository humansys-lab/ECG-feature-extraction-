#!/usr/bin/env python3
"""Compatibility entry point for the recoverable ECGAgent batch runner."""
from ecgagent.batch import main


if __name__ == "__main__":
    raise SystemExit(main())

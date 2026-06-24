"""Thin backward-compatible wrapper → delegates to inference/ + cli/infer.py.

This file is kept for backward compatibility only.
For new code, use:
  - inference.api.ModelInference   (API)
  - inference.scan.scan_models     (model discovery)
  - cli/infer                      (REPL)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from cli.infer import main, _parse_chain, _random_chunk  # noqa: F401

# Re-export for backward compat
from inference.scan import scan_models as _scan_models, parse_key as _parse_key
from inference.api import ModelInference

if __name__ == "__main__":
    main()

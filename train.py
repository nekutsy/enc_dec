"""Thin backward-compatible wrapper → delegates to cli/train.py.

For new code, use:  cli.train
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from cli.train import main

if __name__ == '__main__':
    main()

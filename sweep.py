"""Thin backward-compatible wrapper → delegates to cli/sweep.py.

For new code, use:  cli.sweep
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from cli.sweep import main

if __name__ == '__main__':
    main()

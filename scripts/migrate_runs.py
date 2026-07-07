#!/usr/bin/env python3
"""Migrate existing plain-hash run dirs to {hash}-{model_name} format.

Adds a symlink old_name → new_name for backward compat, so any code
that looks up a run by plain hash still finds the directory.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

SESSIONS_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent / 'sessions'
RUNS_DIR = SESSIONS_DIR / 'runs'


def main(dry_run: bool = True):
    if not RUNS_DIR.is_dir():
        print('No sessions/runs/ directory')
        return

    migrated = 0
    skipped = 0

    for entry in sorted(RUNS_DIR.iterdir()):
        if not entry.is_dir():
            continue

        name = entry.name

        # Already has model_name suffix? (12-char prefix + '-' + something)
        if len(name) > 12 and name[12] == '-':
            skipped += 1
            continue

        # Pure 12-char hash? Read meta.json for model_name
        meta_path = entry / 'meta.json'
        if not meta_path.exists():
            print(f'  ⚠ {name}: no meta.json — skip')
            skipped += 1
            continue

        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f'  ⚠ {name}: bad meta.json ({e}) — skip')
            skipped += 1
            continue

        model_name = meta.get('model_name', '')
        if not model_name:
            # Try to reconstruct from meta
            mc = meta.get('model_config', {})
            sizes = meta.get('layer_sizes', [])
            if sizes and mc:
                shape = mc.get('shape', 'rect')[:4]
                seq_len = mc.get('seq_len', '?')
                mid = len(sizes) // 2
                n_hidden = mid - 1 if mid > 0 else 0
                input_dim = sizes[0] if sizes else None
                hidden_dim = sizes[1] if len(sizes) > 1 else None
                b_val = round(hidden_dim / input_dim, 4) if input_dim and hidden_dim else '?'
                model_name = f'{shape}_s{seq_len}_n{n_hidden}_b{b_val}'
            else:
                model_name = 'unknown'

        new_name = f'{name}-{model_name}'
        new_path = RUNS_DIR / new_name

        if new_path.exists():
            print(f'  ⚠ {name}: target {new_name} already exists — skip')
            skipped += 1
            continue

        action = '[DRY RUN]' if dry_run else '→'
        print(f'  {action} {name}  →  {new_name}')

        if dry_run:
            migrated += 1
            continue

        # Rename
        entry.rename(new_path)

        # Symlink: old name → new name (backward compat for plain-hash lookups)
        os.symlink(new_name, entry, target_is_directory=True)

        migrated += 1

    print(f'\n{"DRY RUN — no changes made. Use --apply to execute." if dry_run else ""}')
    print(f'Migrated: {migrated}  Skipped: {skipped}')


if __name__ == '__main__':
    dry = '--apply' not in sys.argv
    main(dry_run=dry)

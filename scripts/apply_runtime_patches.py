"""Apply rAvI's tracked patches to the pinned Stable Audio 3 checkout. Idempotent.

Each patch in ravi/local/runtime_patches/ is skipped when already applied, applied when it
applies cleanly, and otherwise reported as an error (the checkout has unexpected edits).
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / '.local-runtime/stable-audio-3'
PATCHES = sorted((ROOT / 'ravi/local/runtime_patches').glob('*.patch'))


def git_apply(*args) -> bool:
    return subprocess.run(['git', '-C', str(RUNTIME), 'apply', *map(str, args)],
                          capture_output=True).returncode == 0


def main() -> None:
    if '--reverse' in sys.argv[1:]:
        # Before switching the pinned commit: take our patches back out so git can check out cleanly.
        for patch in reversed(PATCHES):
            if git_apply('--reverse', '--check', patch):
                if not git_apply('--reverse', patch):
                    sys.exit(f'could not reverse {patch.name}')
                print(f'reversed: {patch.name}')
        return
    for patch in PATCHES:
        if git_apply('--reverse', '--check', patch):
            print(f'already applied: {patch.name}')
        elif git_apply('--check', patch):
            if not git_apply(patch):
                sys.exit(f'{patch.name} passed --check but failed to apply')
            print(f'applied: {patch.name}')
        else:
            sys.exit(f'{patch.name} does not apply to {RUNTIME}; inspect local edits there')


if __name__ == '__main__':
    main()

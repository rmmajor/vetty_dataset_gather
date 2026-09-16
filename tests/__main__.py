"""Dependency-free runner: `python -m tests [module_substring ...]` runs every test_* function."""

import importlib
import pkgutil
import sys
import traceback

import tests
from tests import FIRST_PARTY


def load(name):
    """A test module, or None when only an OPTIONAL third-party dependency is missing."""
    try:
        return importlib.import_module(f"tests.{name}")
    except ModuleNotFoundError as error:
        if (error.name or "").split(".")[0] in FIRST_PARTY:
            raise
        print(f"SKIP  {name} (missing dependency: {error.name})")
        return None


def main(filters):
    total = failed = 0
    skipped = []
    for info in pkgutil.iter_modules(tests.__path__):
        if not info.name.startswith("test_") or (filters and not any(f in info.name for f in filters)):
            continue
        module = load(info.name)
        if module is None:
            skipped.append(info.name)
            continue
        for name in sorted(dir(module)):
            if not name.startswith("test_") or not callable(getattr(module, name)):
                continue
            total += 1
            try:
                getattr(module, name)()
                print(f"ok    {info.name}.{name}")
            except Exception:
                failed += 1
                print(f"FAIL  {info.name}.{name}")
                traceback.print_exc()
    print(f"{total - failed}/{total} passed" + (f" ({len(skipped)} skipped: {', '.join(skipped)})" if skipped else ""))
    return 1 if failed or not total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Dependency-free runner: `python -m tests [module_substring ...]` runs every test_* function."""

import importlib
import pkgutil
import sys
import traceback

import tests


def main(filters):
    total = failed = 0
    for info in pkgutil.iter_modules(tests.__path__):
        if not info.name.startswith("test_") or (filters and not any(f in info.name for f in filters)):
            continue
        module = importlib.import_module(f"tests.{info.name}")
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
    print(f"{total - failed}/{total} passed")
    return 1 if failed or not total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

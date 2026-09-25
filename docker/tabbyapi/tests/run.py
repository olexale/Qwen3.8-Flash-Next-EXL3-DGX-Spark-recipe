#!/usr/bin/env python3
"""Minimal test runner for the image's patches (the image has no pytest).

Runs every `test_*` function in the `test_*.py` files next to this script (coroutines via
asyncio.run), prints one line per test and a summary, and exits non-zero if any failed.
Arguments are substrings to select tests by "file::function" name.

Run inside the image: docker/tabbyapi/tests/run_tests.sh
"""
import asyncio, importlib.util, inspect, pathlib, sys, time, traceback

here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(here))
select = sys.argv[1:]
passed, failed = 0, []
for path in sorted(here.glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        failed.append(f"{path.name} (import)")
        print(f"FAIL {path.name}: import error\n{traceback.format_exc()}", flush=True)
        continue
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("test_") or fn.__module__ != mod.__name__:
            continue
        label = f"{path.name}::{name}"
        if select and not any(s in label for s in select):
            continue
        t = time.time()
        try:
            r = fn()
            if inspect.iscoroutine(r):
                asyncio.run(r)
            passed += 1
            print(f"PASS {label} ({time.time() - t:.2f}s)", flush=True)
        except Exception:
            failed.append(label)
            print(f"FAIL {label}\n{traceback.format_exc()}", flush=True)
print(f"\n{passed} passed, {len(failed)} failed" + (": " + ", ".join(failed) if failed else ""), flush=True)
sys.exit(1 if failed else 0)

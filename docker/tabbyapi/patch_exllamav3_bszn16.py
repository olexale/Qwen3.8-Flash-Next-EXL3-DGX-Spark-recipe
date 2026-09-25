"""Fused decode kernels for up to 16 rows (EXL3_MOE_BSZN_MAX=16; default 8, as before).

The fork's fused decode MoE path (BC_BlockSparseMLP.run_bszN: the two exl3_moe_coop launches
per layer, and the shared expert's multi-row graph, BC_GatedMLP.run_bszN) takes batches of
up to MAX_BSZN = 8 rows. A verify of 9+ rows (more than 7 drafted tokens for one session, or
two sessions drafting 4+ each) falls through to the prefill-style fused MoE kernel, which
costs ~45 ms more per round on this model (a round verifying 11 rows ~126 ms, 16 rows
~147 ms, against ~65 ms at 8).

This builds the extension with MAX_BSZN 16 (the coop kernels bound their slots at 256 =
16 x top-k 16, and size their counters and scratch from the Python-side buffers), and makes
the Python-side limit a run-time setting: EXL3_MOE_BSZN_MAX (default 8, max 16). With the
default, every buffer the kernels use and every launch is exactly as before (split-k is
off, so the launch geometry does not depend on the scratch size), so outputs are
bit-identical; with 16, batches of 9-16 rows take the decode kernels.

Usage: python3 patch_exllamav3_bszn16.py <exllamav3 source dir>   (before the extension build)
       python3 patch_exllamav3_bszn16.py --installed [package dir]  (Python side, after install)
Exits non-zero if an anchor is missing.
"""
import importlib.util, pathlib, sys

PY_BSM = ("MAX_BSZN = 8  # must match MAX_BSZN in exllamav3_ext/libtorch/blocksparse_mlp.h",
          "import os as _os\n"
          "# must not exceed MAX_BSZN in exllamav3_ext/libtorch/mlp.h (16, patch_exllamav3_bszn16.py)\n"
          "MAX_BSZN = max(1, min(16, int(_os.environ.get(\"EXL3_MOE_BSZN_MAX\", \"8\"))))")
PY_MLP = ("MAX_BSZN = 8  # must match MAX_BSZN in exllamav3_ext/libtorch/mlp.h and block_sparse_mlp.py",
          "import os as _os\n"
          "# must not exceed MAX_BSZN in exllamav3_ext/libtorch/mlp.h (16, patch_exllamav3_bszn16.py)\n"
          "MAX_BSZN = max(1, min(16, int(_os.environ.get(\"EXL3_MOE_BSZN_MAX\", \"8\"))))")
CPP_H = ("#define MAX_BSZN 8   // must match MAX_BSZN in blocksparse_mlp.h / BlockSparseMLP.py",
         "#define MAX_BSZN 16  // upper bound; the Python side picks the limit (EXL3_MOE_BSZN_MAX, patch_exllamav3_bszn16.py)")

def edit(p, old, new):
    s = p.read_text()
    if s.count(old) != 1:
        sys.exit(f"patch_exllamav3_bszn16: anchor found {s.count(old)}x (want 1) in {p}:\n{old}")
    p.write_text(s.replace(old, new))

if len(sys.argv) > 1 and sys.argv[1] == "--installed":
    root = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else \
        pathlib.Path(importlib.util.find_spec("exllamav3").submodule_search_locations[0])
    edit(root / "modules/block_sparse_mlp.py", *PY_BSM)
    edit(root / "modules/mlp.py", *PY_MLP)
else:
    src = pathlib.Path(sys.argv[1]) / "exllamav3"
    edit(src / "exllamav3_ext/libtorch/mlp.h", *CPP_H)
    edit(src / "modules/block_sparse_mlp.py", *PY_BSM)
    edit(src / "modules/mlp.py", *PY_MLP)
print("patch_exllamav3_bszn16: applied")

"""Turn exllamav3's fused MoE kernel back on for packs whose experts share one K.

BlockSparseMLP.load_local() sets support_fused, which gives prefill the fused
multi-expert kernel (exl3_moe: every expert with up to 256 rows in a few
launches). The fork's per-K-group commit (785f206) moved that assignment into
the branch for mixed-K packs, so a uniform pack like this one never gets it:
support_fused stays False and prefill runs every expert with <= 32 rows as its
own graph launch, ~475 experts x 48 layers on a 600-token prompt. This moves
the block back to where upstream has it, next to the MultiLinear setup.

EXL3_MOE_FUSED_UNIFORM=0 at run time restores the fork's behaviour.

Usage: python3 patch_exllamav3_fused_moe.py [exllamav3 package dir]
Run once at image build time; exits non-zero if the anchor is missing.
"""
import importlib.util, pathlib, sys
if len(sys.argv) > 1:
    root = pathlib.Path(sys.argv[1])
else:
    # Locate the package without importing it (the import loads the CUDA extension)
    root = pathlib.Path(importlib.util.find_spec("exllamav3").submodule_search_locations[0])
rel = "modules/block_sparse_mlp.py"
old = """
            # Enable fully fused kernel if possible (uniform mcg or mul1 codebook across gate/up/down,
            # and an activation the fused kernel implements)
            if self.multi_up is not None:
                cbs = (
                    self.multi_gate.q_cb() if self.gated else self.multi_up.q_cb(),
                    self.multi_up.q_cb(),
                    self.multi_down.q_cb(),
                )
                self.support_fused = (
                    cbs[0] == cbs[1] == cbs[2] and cbs[0] in ((True, False), (False, True)) and
                    self.support_quant_paths
                )
"""
new = """
        # Enable fully fused kernel if possible (uniform mcg or mul1 codebook across gate/up/down,
        # and an activation the fused kernel implements). Patched out of the mixed-K branch
        # (patch_exllamav3_fused_moe.py); EXL3_MOE_FUSED_UNIFORM=0 keeps it off
        if self.multi_up is not None and os.environ.get("EXL3_MOE_FUSED_UNIFORM", "1") != "0":
            cbs = (
                self.multi_gate.q_cb() if self.gated else self.multi_up.q_cb(),
                self.multi_up.q_cb(),
                self.multi_down.q_cb(),
            )
            self.support_fused = (
                cbs[0] == cbs[1] == cbs[2] and cbs[0] in ((True, False), (False, True)) and
                self.support_quant_paths
            )
"""
p = root / rel
s = p.read_text()
if s.count(old) != 1 or "\nimport os" not in s:
    sys.exit(f"patch_exllamav3_fused_moe: anchor not found exactly once in {rel}")
p.write_text(s.replace(old, new))
print("patched", rel)

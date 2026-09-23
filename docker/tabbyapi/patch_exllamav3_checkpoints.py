"""Keep recurrent-state checkpoints on the device (GB10 has unified memory).

The generator checkpoints every recurrent layer's state (36 GatedDeltaNet layers
plus the PLE layers on this model) every 2,048 tokens near the end of a prompt, so
a later turn can resume from the cached prefix. Stock exllamav3 stashes each state
with .cpu(): a blocking copy into freshly allocated pageable host memory. On a
discrete GPU that frees VRAM; on GB10 the "host" and "device" are the same memory,
so it only costs time: about 0.9 s on a 600-token prompt, 2.2 s on 4k (profiled
2026-09-23). .clone() keeps the checkpoint on the device. The generator's
recurrent_cache_size still bounds the total.

Run once at image build time; exits non-zero if an anchor is missing.
"""
import importlib.util, pathlib, sys
# Locate the package without importing it (the import loads the CUDA extension)
root = pathlib.Path(importlib.util.find_spec("exllamav3").submodule_search_locations[0])
edits = {
    "modules/gated_delta_net.py": [
        ("            self.recurrent_state[slot, :1].cpu(),\n            self.conv_state[slot, :, :cdim].cpu()\n",
         "            self.recurrent_state[slot, :1].clone(),\n            self.conv_state[slot, :, :cdim].clone()\n"),
    ],
    "modules/ple.py": [
        ("return (self.conv_state[slot, :, :self.win].cpu(), self.id_state[slot, :self.ctx].cpu())",
         "return (self.conv_state[slot, :, :self.win].clone(), self.id_state[slot, :self.ctx].clone())"),
    ],
}
for rel, reps in edits.items():
    p = root / rel
    s = p.read_text()
    for old, new in reps:
        if s.count(old) != 1:
            sys.exit(f"patch_exllamav3_checkpoints: anchor not found exactly once in {rel}")
        s = s.replace(old, new)
    p.write_text(s)
    print("patched", rel)

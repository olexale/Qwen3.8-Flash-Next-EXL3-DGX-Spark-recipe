#!/bin/sh
# Drop the model's pages from the page cache, then start TabbyAPI.
#
# On GB10 cached file pages count against GPU-allocatable memory, and a load
# after an earlier one (a crash plus --restart, or docker restart) fails if the
# pack is still cached. Same fadvise as scripts/exl3_native/tuning/drop-model-cache.sh;
# it needs no root and works through a read-only mount.
python3 - <<'EOF'
import os
n = 0
for root, _, files in os.walk("/models", followlinks=True):
    for name in files:
        try:
            fd = os.open(os.path.join(root, name), os.O_RDONLY)
        except OSError:
            continue
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
        n += 1
print(f"entrypoint: dropped page cache for {n} model files", flush=True)
EOF
exec python3 main.py "$@"

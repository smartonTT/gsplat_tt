"""Task #409: Ethernet dispatch (12x10 = 120 compute cores) is the p150 default.

Stock tt-metal cannot open ETH dispatch on a p150 (14-core yaml lists, 24 KB idle-ERISC
.ld), so the run points TT_METAL_RUNTIME_ROOT at an overlay from opt/eth/make_overlay.sh
(no tt-metal edit or rebuild) with a JIT cache of its own. setup() runs before the device
opens and sets GSPLAT_TT_DISPATCH to what it decided:
  - GSPLAT_TT_DISPATCH=worker: worker, nothing else touched (override).
  - card without ETH cores (p100a) or unknown: worker.
  - p150/p300, or GSPLAT_TT_DISPATCH=eth: reuse an overlay already in TT_METAL_RUNTIME_ROOT
    (opt/eth/env.sh), else make one at GSPLAT_TT_ETH_OVERLAY (default <repo>/tmp/ttm-eth12)
    from TT_METAL_HOME. If that fails, fall back to worker with a log line (an explicit
    eth is kept as asked).
render/host/device_state.cpp resolves an unset/auto GSPLAT_TT_DISPATCH the same way, so
other entry points (the viewer) stay on worker unless they run with an overlay.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Callable, MutableMapping

MARK = ".gsplat-eth-overlay"
N_ETH = 12  # live ETH cores on a p150 (ETH harvesting mask 0x120)


def card_type(device_id: int = 0, root: str = "/sys/class/tenstorrent") -> str:
    """tt-kmd card type of /dev/tenstorrent/<id> ("p150b", "p100a", ...); "" if unknown."""
    try:
        return Path(f"{root}/tenstorrent!{device_id}/tt_card_type").read_text().strip().lower()
    except OSError:
        return ""


def card_has_eth(card: str) -> bool:
    return card.startswith(("p150", "p300"))


def overlay_info(path: str | None) -> dict | None:
    """The overlay marker's key=value lines, or None if `path` is not an overlay."""
    if not path:
        return None
    try:
        text = (Path(path) / MARK).read_text()
    except OSError:
        return None
    return dict(ln.split("=", 1) for ln in text.splitlines() if "=" in ln)


def setup(env: MutableMapping[str, str], repo_root: Path, card: str | None = None,
          run: Callable = subprocess.run, log: Callable[[str], None] = print) -> str:
    """Pick the dispatch mode for this run and export what it needs into `env`."""
    req = env.get("GSPLAT_TT_DISPATCH", "").strip().lower()
    if req == "worker":
        log("[eth] GSPLAT_TT_DISPATCH=worker: worker dispatch (override)")
        return "worker"
    if req not in ("", "auto", "eth"):
        return req  # device_state.cpp logs the unknown value and uses worker
    forced = req == "eth"
    card = card_type() if card is None else card
    if not forced and not card_has_eth(card):
        env["GSPLAT_TT_DISPATCH"] = "worker"
        log(f"[eth] card {card or 'unknown'}: worker dispatch (ETH default is p150 only)")
        return "worker"

    def fallback(why: str) -> str:
        if forced:
            log(f"[eth] WARNING: {why}; GSPLAT_TT_DISPATCH=eth kept as asked")
            return "eth"
        env["GSPLAT_TT_DISPATCH"] = "worker"
        log(f"[eth] ETH overlay setup failed ({why}); falling back to worker dispatch")
        return "worker"

    if overlay_info(env.get("TT_METAL_RUNTIME_ROOT")) is not None:
        ov = env["TT_METAL_RUNTIME_ROOT"]  # opt/eth/env.sh already set it and the caches
        log(f"[eth] card {card or 'unknown'}: eth dispatch, overlay {ov} (from env)")
    else:
        src = env.get("TT_METAL_HOME", "")
        if not src:
            return fallback("TT_METAL_HOME unset")
        ov = env.get("GSPLAT_TT_ETH_OVERLAY") or str(repo_root / "tmp" / f"ttm-eth{N_ETH}")
        info = overlay_info(ov)
        # Reuse a matching overlay: rebuilding swaps the dir, which a concurrent run may use.
        if info is None or info.get("src") != os.path.realpath(src) or info.get("n_eth") != str(N_ETH):
            r = run(["bash", str(repo_root / "opt" / "eth" / "make_overlay.sh"), src, ov, str(N_ETH)],
                    capture_output=True, text=True)
            if r.returncode != 0:
                tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or [""]
                return fallback(f"make_overlay.sh rc={r.returncode}: {tail[0]}")
            if overlay_info(ov) is None:
                return fallback(f"no {MARK} in {ov}")
        cache = env.get("GSPLAT_TT_ETH_CACHE") or ov + "-cache"
        # ELFs linked with the overlay's 32 KB idle-ERISC bound must not mix with another cache.
        env["TT_METAL_RUNTIME_ROOT"] = ov
        env["TT_METAL_CACHE"] = cache + "/prod"
        env["TT_METAL_CACHE_RENDER"] = cache + "/render"
        log(f"[eth] card {card or 'unknown'}: eth dispatch, overlay {ov}, JIT cache {cache}")
    env["GSPLAT_TT_DISPATCH"] = "eth"
    return "eth"

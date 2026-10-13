# A/B gate noise floor (#656)

Why: #647 ran 4 rotated rounds of `render/run.py --back-to-back` on two builds whose
kernels were byte-identical (bh-33 JIT caches ttmc-t647abl and ttmc-t647raw, same 21
hashes) and still saw a 0.04-0.10 ms/view same-sign delta (#654,
docs/absmeans-t647/t654-shelve.md). Same-sign over 4 rounds is not proof of a real effect.

## Rules for every A/B gate

1. **JIT-hash check per arm.** After the run, list each arm's JIT cache and check that
   the kernel the gate edits has different hashes between the arms:

       # on the box, per arm
       find <arm cache> -maxdepth 6 -path '*/kernels/*/*' -type d > <arm>.jit.txt
       # locally (or on the box)
       python3 opt/ab_jit_check.py --arm base=base.jit.txt --arm cand=cand.jit.txt \
           --kernel <edited kernel name>

   Exit 1 (`JIT-HASH FAIL`) means the gate is A/A for that kernel: the delta is not a
   result, the gate fails, and the hand-off says so. A host-only lever (no kernel edit)
   names no kernel and states that in the result instead.
2. **Noise rule.** A b2b delta under ~0.1 ms/view from 4 rounds is noise, unless a
   same-day A/A run (two arms with identical JIT hashes, same box, same rotation) shows
   a tighter spread; then the floor is that A/A spread. Do not keep or shelve a lever
   on such a delta alone.

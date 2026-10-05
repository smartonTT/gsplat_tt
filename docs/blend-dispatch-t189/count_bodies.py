#!/usr/bin/env python3
"""t189: count issued instructions per blend body in a TRISC1 objdump.
  count_bodies.py out/trisc1-default.dis
A playback ttreplay S,N,0,0 adds N issued SFPU ops (a recording one executes inline)."""
import re, sys
from collections import Counter, defaultdict
bodies = defaultdict(Counter)
cur = None
for line in open(sys.argv[1]):
    m = re.match(r'^[0-9a-f]+ <(.*blend_(pair|one)_body.*)>:', line)
    if m:
        cur = m.group(1); continue
    if re.match(r'^[0-9a-f]+ <', line) and 'LBB' not in line:
        cur = None; continue
    if cur is None:
        continue
    m = re.match(r'^\s+[0-9a-f]+:\s+[0-9a-f]+\s+(\S+)\s*(.*)', line)
    if not m:
        continue
    op, args = m.groups()
    c = bodies[cur]
    if op == 'ttreplay':
        a = [int(x) for x in args.split(',')]
        if a[3] == 0:  # load_mode 1 records while executing; only playback adds ops
            c['replayed'] += a[1]
    elif op.startswith('sfp'):
        c['sfpu'] += 1; c[op] += 1
    else:
        c['risc'] += 1
    if op == 'ret':
        cur = None
kinds = defaultdict(list)
for name, c in bodies.items():
    kinds['pair' if name.count(',') or '3ul' in name else 'single'].append(c)
for name, c in sorted(bodies.items())[:4]:
    print(f"{name}: sfpu {c['sfpu']} replayed {c['replayed']} issued {c['sfpu']+c['replayed']} "
          f"nop {c['sfpnop']} load {c['sfpload']} store {c['sfpstore']} risc {c['risc']}")
print(f'bodies found: {len(bodies)}')

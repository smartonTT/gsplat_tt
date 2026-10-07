#!/bin/bash
# t370: host-only probe (no device use). Copies copy_bench.c to <host>:/tmp, builds and runs it
# twice, and records CPU model/clock, load, the top CPU users and the card's PCIe link.
#   host_probe.sh <host>   (bh-30 under `ttp lock viewer`, yyzo-bh-04 under `ttp lock p100`)
set -u
H=$1; D=$(cd "$(dirname "$0")" && pwd)
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=20 "$H")
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes "$D/copy_bench.c" "$H:/tmp/t370_copy_bench.c" || exit 3
"${SSH[@]}" 'set -u; echo "== $(hostname) $(date -u +%FT%TZ)"
grep -m1 "model name" /proc/cpuinfo; nproc
for c in 0 1; do echo "cpu$c cur=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq 2>/dev/null) max=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/cpuinfo_max_freq 2>/dev/null) gov=$(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_governor 2>/dev/null)"; done
uptime
for d in /sys/class/tenstorrent/tenstorrent!*/device; do echo "$d link $(cat $d/current_link_speed) x$(cat $d/current_link_width) numa $(cat $d/numa_node)"; done
grep -E "^(HugePages_Total|HugePages_Free|AnonHugePages)" /proc/meminfo; cat /sys/kernel/mm/transparent_hugepage/enabled
echo "-- top cpu users"; ps -eo pcpu,pid,user,comm --sort=-pcpu | head -6
cc -O2 -o /tmp/t370_copy_bench /tmp/t370_copy_bench.c && for r in 1 2; do echo "-- run $r"; /tmp/t370_copy_bench; done
rm -f /tmp/t370_copy_bench /tmp/t370_copy_bench.c'

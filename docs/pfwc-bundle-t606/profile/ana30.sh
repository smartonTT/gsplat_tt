#!/bin/bash
# t608: per-chunk pfwc summaries for out30 (views 5:30), 4 workers
cd "$(dirname "$0")/out30" || exit 1
ls dev-*.csv.gz | xargs -P 4 -I{} sh -c 't=$(basename {} .csv.gz); t=${t#dev-}; python3 ../pfwc608.py {} --label $t > prof-$t.txt 2>&1; python3 ../progwin608.py {} > progwin-$t.txt 2>&1'

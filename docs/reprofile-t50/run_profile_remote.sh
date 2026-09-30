set -u
T=/localdev/smarton/gstt2-t50
export TTW_DEVRUN=1 TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=P100
export TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t50
source /localdev/smarton/gstt2/.venv/bin/activate
cd $T || exit 1
echo "== SHA $(cat SHA) build $(date +%T)"
mkdir -p tmp; rm -f render/render_clean*.so
(cmake -G Ninja -S render -B render/build-tt -DCMAKE_BUILD_TYPE=Release > tmp/cfg.log 2>&1 && cmake --build render/build-tt -j 16 > tmp/build.log 2>&1) || { tail -30 tmp/cfg.log tmp/build.log; exit 1; }
md5sum render/render_clean*.so
for r in 1 2 3; do echo "=== timing r$r $(date +%T)"; timeout 300 python3 render/run.py --no-ref --iter-dir t50-time 2>&1 | grep -E '^(SUMMARY|STAGES|SORT_STAGES)|rror|hang|REFUS' | head -10; done
unset TT_METAL_CACHE_RENDER
export GSTT2_REPO=$T TT_METAL_CACHE_RENDER=/localdev/smarton/.cache/ttmc-gstt2-t50-prof
csvs=""
for ab in 0:10 10:20 20:30; do echo "== tracy $ab $(date +%T)"; timeout 480 bash opt/profiler/capture_tracy.sh ttw-t50 $ab 2>&1 | grep -E 'capture_tracy\] (OK|FAIL|DONE|device profiler CSV)|SUMMARY|rror' | head; csvs="$csvs opt/profiler/ttw-t50/chunks/${ab/:/-}/profile_log_device.csv"; done
cd $T; D=opt/profiler/ttw-t50
python3 opt/profiler/stitch_device_csv.py -o $D/profile_log_device.csv $csvs 2>&1 | tail -8
python3 opt/profiler/zone_occupancy.py $D/profile_log_device.csv > $D/zone_occupancy.txt 2>&1; echo "zone_occ rc=$?"
python3 opt/profiler/program_gaps.py $D/profile_log_device.csv > $D/program_gaps.txt 2>&1; echo "gaps rc=$?"
python3 opt/profiler/analyze_zones.py $D/profile_log_device.csv 30 > $D/zones.txt 2>&1; echo "zones rc=$?"
echo "== end $(date +%T)"

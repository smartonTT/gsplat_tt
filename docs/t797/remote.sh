#!/bin/bash
# t797: one step on the measurement box (bh-49 p150a), run by docs/t797/drive.sh under `ttp lock p100`.
#   remote.sh build                   build-avx2-tt: gsplat_cpu_tests + _gsplat_cpu; gsplat_cpu_tests '[project]'
#   remote.sh pytest                  T0 dumps current? (else regenerate in place, kept in tmp/t797/dumps);
#                                     then the train / T2 / blend-backward unit tests on the box venv
#   remote.sh run <round> <arm>...    arms: cand (defaults, GSPLAT_TT_JCLAMP=1) / base (GSPLAT_TT_JCLAMP=0),
#                                     prefixed o (orbit b2b, cameras_v2 30 views, 20 passes), n (train NN
#                                     b2b, 169 views, 6 passes), h (orbit hero with the CPU ref: golden +
#                                     hero_vs_cpu), d (_DSC8679 1237x822, cameras_test view 0)
# Each arm has its own eth JIT cache (GSPLAT_TT_ETH_CACHE); the cand cache must hold PFWC_JCLAMP, base must not.
set -u
export TT_METAL_HOME=/localdev/smarton/tt-metal TT_METAL_RUNTIME_ROOT=/localdev/smarton/tt-metal
export TT_METAL_ARCH_NAME=blackhole MESH_DEVICE=${MESH_DEVICE:-P150} TTW_DEVRUN=1 PYTHONDONTWRITEBYTECODE=1
unset GSPLAT_TT_DISPATCH GSPLAT_TT_ETH_OVERLAY GSPLAT_TT_ETH_CACHE TT_METAL_CACHE TT_METAL_CACHE_RENDER \
  GSPLAT_PER_VIEW_STAGES GSPLAT_TT_HOST_PROFILE GSPLAT_TT_JCLAMP
T=${T:?T}; cd "$T" || exit 1; source .venv/bin/activate
C=/localdev/smarton/.cache/ttmc-t797; S=$T/tmp/t797; mkdir -p $S
cmd=${1:?cmd}; shift
case $cmd in
build)
  (cmake -G Ninja -S src -B build-avx2-tt -DCMAKE_BUILD_TYPE=Release > $S/cpu_cfg.log 2>&1 \
    && nice -n 19 cmake --build build-avx2-tt --target gsplat_cpu_tests _gsplat_cpu -j 16 > $S/cpu_build.log 2>&1)
  rc=$?; echo "cpu build rc=$rc"; [ $rc = 0 ] || { tail -n 40 $S/cpu_cfg.log $S/cpu_build.log; exit 5; }
  md5sum backends/cpu_cpp/_gsplat_cpu*.so
  timeout 300 build-avx2-tt/tests/gsplat_cpu_tests '[project]' > $S/cpu_tests_project.log 2>&1
  rc=$?; echo "gsplat_cpu_tests [project] rc=$rc"; tail -4 $S/cpu_tests_project.log; exit $rc ;;
pytest)
  PT=(python3 -m pytest -q -p no:cacheprovider -rs)
  timeout 900 "${PT[@]}" tests/unit/test_train_oracle.py -k dumps_are_current > $S/pytest-dumps.log 2>&1
  rc=$?; echo "dumps_are_current rc=$rc"; tail -3 $S/pytest-dumps.log
  if [ $rc != 0 ]; then
    timeout 900 python3 -m gsplat.train_oracle --out benchmarks/train_oracle > $S/regen.log 2>&1; echo "regen rc=$?"
    mkdir -p $S/dumps && cp benchmarks/train_oracle/t0_sh*_64x64_n50.npz $S/dumps/ && md5sum $S/dumps/*.npz
  fi
  timeout 2400 "${PT[@]}" tests/unit/test_train_oracle.py tests/unit/test_t2_param_backward.py \
    tests/unit/test_blend_backward_dev.py tests/unit/test_blend_backward_ref.py tests/unit/test_train_loop_dev.py \
    tests/unit/test_train_loss.py tests/unit/test_train_opc_gather_t727.py tests/unit/test_train_opc_ring3_t755.py \
    tests/unit/test_train_opc_side_t741.py tests/unit/test_train_opc_writer_t730.py tests/unit/test_train_step_dev.py \
    > $S/pytest.log 2>&1
  rc=$?; echo "pytest rc=$rc"; grep -E '^(FAILED|ERROR|SKIPPED)' $S/pytest.log | cut -c1-240 | head -20; tail -1 $S/pytest.log; exit $rc ;;
run)
  r=${1:?round}; shift; fail=0
  for a in "$@"; do
    arm=${a:1}; kind=${a:0:1}; tag=r$r-$a; env=()
    case $arm in cand) ;; base) env=(GSPLAT_TT_JCLAMP=0) ;; *) echo "bad arm $a"; exit 2 ;; esac
    case $kind in
      o) args=(--no-ref --back-to-back) ;;
      n) args=(--no-ref --back-to-back --b2b-passes 6 --cameras benchmarks/cameras_train_nn.json) ;;
      h) args=(--view-range 0:1) ;;
      d) args=(--no-ref --view-range 0:1 --cameras benchmarks/cameras_test.json) ;;
      *) echo "bad kind $a"; exit 2 ;;
    esac
    echo "=== $tag $(cut -c1-8 SHA) ${env[*]+${env[*]}} ${args[*]} $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
    env ${env[@]+"${env[@]}"} GSPLAT_TT_ETH_CACHE=$C-$arm timeout ${RUN_TIMEOUT:-280} \
      python3 render/run.py --iter-dir t797-$tag "${args[@]}" > $S/run-$tag.log 2>&1
    rc=$?; echo "run rc=$rc"
    grep -E "^(B2B |SUMMARY)|\[eth\]|TTW_TIMING (ms_view|b2b_ms_frame_median)=|Traceback|TT_THROW|TT_FATAL|error:" $S/run-$tag.log | cut -c1-400 | head -10
    if [ $rc = 124 ] || [ $rc = 137 ]; then echo "HANG in $tag: tt-smi -r"; tt-smi -r > $S/reset-$tag.log 2>&1; echo "reset rc=$?"; exit 124; fi
    [ $rc = 0 ] || fail=1
    if [ -f tmp/t797-$tag/hero_clean.png ]; then
      cp tmp/t797-$tag/hero_clean.png $S/hero-$tag.png; echo "HERO_MD5 $tag $(md5sum < $S/hero-$tag.png | cut -c1-32)"
      for f in hero_ref.png hero_diff10.png hero_diff10_cpu.png; do [ -f tmp/t797-$tag/$f ] && cp tmp/t797-$tag/$f $S/${f%.png}-$tag.png; done
    fi
    rm -rf tmp/t797-$tag
    n=$(grep -rl 'PFWC_JCLAMP' $C-$arm 2>/dev/null | wc -l)
    echo "LEVER $tag files_with_PFWC_JCLAMP=$n"
    case $arm in cand) [ "$n" -gt 0 ] || { echo "INERT $tag"; fail=1; } ;; base) [ "$n" = 0 ] || { echo "BASE_NOT_OFF $tag"; fail=1; } ;; esac
  done
  echo "=== done $r $(date +%T)"; exit $fail ;;
*) echo "bad cmd $cmd"; exit 2 ;;
esac

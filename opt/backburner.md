# Backburner

Issues deferred until we are at the 1ms/frame goal.

- TT-METAL ShmResourceTracker::cleanup_all double-free at process exit — likely device_state holders racing with tt-metal atexit handler. Doesn't affect render correctness or measurements. Investigate after 1ms/frame goal.
- tt-008c cov2d precision: device kernel produces cov2d (a, b, c) PSNR-equivalent to CPU (47.09 dB) but with ~ulp-scale numerical drift that defeats microblock_drop_pct cull (0% vs 12.3% iter-09). Blend median wall time UNCHANGED so immaterial for ms/frame; investigate root cause (SFPU recip APPROX flag? math_approx_mode propagation? arithmetic op order?) after 1ms/frame goal.

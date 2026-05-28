# Backburner

Issues deferred until we are at the 1ms/frame goal.

- TT-METAL ShmResourceTracker::cleanup_all double-free at process exit — likely device_state holders racing with tt-metal atexit handler. Doesn't affect render correctness or measurements. Investigate after 1ms/frame goal.

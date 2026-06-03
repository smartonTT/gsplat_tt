# Metal iter-005+ — perf toward 1 ms/frame

Multi-core LPT dispatch tuning, replay buffer, frame-coherent cov3d reuse, SFPU pipelining.
Regression budget per iter from `opt/plan-amendment-001-metal-port.md`.

Halt when 30-view sum_ms < 30 ms or physical lower bound reached.

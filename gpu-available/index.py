"""
Compare large matrix-multiply throughput on CPU vs Apple MPS (Metal GPU).

Warmup runs prime caches and amortize one-time Metal setup before timed runs.
MPS kernels run asynchronously; torch.mps.synchronize() fences the GPU so timings
reflect real work completion, not just kernel enqueue time.
"""
import statistics
import time

import torch

# Matrix shape (SIZE x SIZE). Larger → more meaningful GPU wins but slower runs.
SIZE = 5000
DTYPE = torch.float32

# Discard first batches so JIT/caches/OS scheduling don’t skew the median.
WARMUP_ROUNDS = 5
TIMED_ROUNDS = 20


def bench_cpu_mm(a: torch.Tensor, b: torch.Tensor, rounds: int) -> list[float]:
    """Time `a @ b` on CPU. CPU matmul completes before perf_counter resumes."""
    times: list[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        _ = a @ b
        times.append(time.perf_counter() - t0)
    return times


def bench_mps_mm(a: torch.Tensor, b: torch.Tensor, rounds: int) -> list[float]:
    """Time `a @ b` on MPS. Sync before start = idle GPU; sync after end = kernels finished."""
    times: list[float] = []
    for _ in range(rounds):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        _ = a @ b
        torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
    return times


# CUDA = NVIDIA; MPS = Apple Silicon GPU via Metal.
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

torch.manual_seed(0)
a_cpu = torch.randn(SIZE, SIZE, dtype=DTYPE)
b_cpu = torch.randn(SIZE, SIZE, dtype=DTYPE)

bench_cpu_mm(a_cpu, b_cpu, WARMUP_ROUNDS)
cpu_times = bench_cpu_mm(a_cpu, b_cpu, TIMED_ROUNDS)
cpu_med = statistics.median(cpu_times)
print(
    f"CPU ({TIMED_ROUNDS} timed runs after {WARMUP_ROUNDS} warm-ups, median): {cpu_med:.4f}s"
)

if torch.backends.mps.is_available():
    # .to("mps") copies weights to unified memory accessible by the GPU.
    a_gpu = a_cpu.to("mps")
    b_gpu = b_cpu.to("mps")

    bench_mps_mm(a_gpu, b_gpu, WARMUP_ROUNDS)
    gpu_times = bench_mps_mm(a_gpu, b_gpu, TIMED_ROUNDS)
    gpu_med = statistics.median(gpu_times)
    print(
        f"MPS ({TIMED_ROUNDS} timed runs after {WARMUP_ROUNDS} warm-ups, median): {gpu_med:.4f}s"
    )

    ratio_cpu_over_mps = cpu_med / gpu_med
    if ratio_cpu_over_mps >= 1:
        print(f"MPS faster than CPU by {ratio_cpu_over_mps:.2f}x (median)")
    else:
        print(f"MPS slower than CPU by {1 / ratio_cpu_over_mps:.2f}x (median)")

# Example output (hardware-dependent):
# CPU (20 timed runs after 5 warm-ups, median): 0.1880s
# MPS (20 timed runs after 5 warm-ups, median): 0.0915s
# MPS faster than CPU by 2.06x (median)

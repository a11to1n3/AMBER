"""Private fused CUDA kernels used by AMBER's GPU execution path.

These helpers are intentionally not part of the public API.  Models continue
to use ``model.gpu().run()``; the execution backend chooses these kernels when
CuPy is active and falls back to ordinary array expressions otherwise.
"""

from __future__ import annotations

from typing import Any


_kernels: dict[tuple[str, str], Any] = {}

_WEALTH_KERNEL = None


def fused_random_walk(x, y, dx, dy, lo: float, hi: float):
    """Apply two bounded movement updates in one elementwise GPU launch."""
    import cupy as cp

    dtype = cp.dtype(x.dtype)
    if dtype not in (cp.dtype("float32"), cp.dtype("float64")):
        return cp.clip(x + dx, lo, hi), cp.clip(y + dy, lo, hi)

    # CuPy's elementwise scalar type resolver is inconsistent for mixed
    # float32 state and Python bounds on some NVRTC versions.  Keep the
    # compact float32 path safe with the backend's native clip expression.
    if dtype == cp.dtype("float32"):
        return cp.clip(x + dx.astype(dtype), lo, hi), cp.clip(y + dy.astype(dtype), lo, hi)

    # CuPy's default RNG emits float64 while compact GPU state may be
    # float32 (notably the fused SIR path).  Normalize the deltas once so the
    # elementwise kernel remains a single typed launch.
    dx = cp.asarray(dx, dtype=dtype)
    dy = cp.asarray(dy, dtype=dtype)

    key = (str(dtype), "random_walk")
    kernel = _kernels.get(key)
    if kernel is None:
        scalar = "float" if dtype == cp.dtype("float32") else "double"
        kernel = cp.ElementwiseKernel(
            f"{scalar} x, {scalar} y, {scalar} dx, {scalar} dy, {scalar} lo, {scalar} hi",
            f"{scalar} out_x, {scalar} out_y",
            (
                f"{scalar} nx = x + dx; {scalar} ny = y + dy; "
                "out_x = nx < lo ? lo : (nx > hi ? hi : nx); "
                "out_y = ny < lo ? lo : (ny > hi ? hi : ny);"
            ),
            f"ambr_fused_random_walk_{scalar}",
        )
        _kernels[key] = kernel

    out_x = cp.empty_like(x)
    out_y = cp.empty_like(y)
    kernel(x, y, dx, dy, dtype.type(lo), dtype.type(hi), out_x, out_y)
    return out_x, out_y


def fused_wealth_transfer(wealth, donor_positions, recipients):
    """Debit a frozen donor set and credit recipients in one GPU kernel.

    ``donor_positions`` must be computed from the step-entry wealth array
    before this function is called.  Keeping eligibility outside the atomic
    kernel prevents a recipient from becoming a donor merely because another
    CUDA thread credited it earlier in the same launch.
    """
    import cupy as cp

    global _WEALTH_KERNEL
    if (
        wealth.dtype != cp.dtype("int64")
        or donor_positions.dtype != cp.dtype("int64")
        or recipients.dtype != cp.dtype("int64")
    ):
        return False
    if int(donor_positions.size) != int(recipients.size):
        raise ValueError("donor_positions and recipients must have equal length")
    if int(donor_positions.size) == 0:
        return True
    if _WEALTH_KERNEL is None:
        _WEALTH_KERNEL = cp.RawKernel(
            r'''
            extern "C" __global__ void wealth_transfer(
                long long* wealth, const long long* donors,
                const long long* recipients, long long n)
            {
                long long i = (long long)blockDim.x * blockIdx.x + threadIdx.x;
                if (i >= n) return;
                long long donor = donors[i];
                // CUDA exposes the 64-bit atomic overload as unsigned.  The
                // bitwise representation is identical for signed int64, so
                // UINT64_MAX is an atomic decrement by one.
                atomicAdd((unsigned long long*)&wealth[donor], 0xffffffffffffffffULL);
                atomicAdd((unsigned long long*)&wealth[recipients[i]], 1ULL);
            }
            ''',
            "wealth_transfer",
        )
    threads = 256
    blocks = (int(donor_positions.size) + threads - 1) // threads
    _WEALTH_KERNEL(
        (blocks,), (threads,),
        (
            wealth,
            donor_positions,
            recipients,
            cp.int64(donor_positions.size),
        ),
    )
    return True

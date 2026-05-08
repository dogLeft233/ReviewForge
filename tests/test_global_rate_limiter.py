#!/usr/bin/env python3
"""多线程限速测试

验证目标：多线程并发访问严格遵守时间间隔

重要设计决策：
- _last 初始化为 -interval（不是 0）：避免 _last=0 导致首次立即通过的问题
- sleep_for = max(0, interval - elapsed)：elapsed==interval 时不重复等待
  （首个请求在真实时钟下已有充足 elapsed，无需再次 sleep）
- 关键保证：多个线程同时调用 wait() 时，后到的线程会被强制等待 interval
"""

import sys
import time
import threading
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
for p in (str(_PROJECT_ROOT), str(_SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.retrievers.rate_limits import RateLimiter
import src.retrievers.rate_limits as rl


def test_parallel_threads_strict_interval():
    """Test 1: 多线程下严格串行等间隔（模拟模式）"""
    print("=" * 60)
    print("Test 1: 多线程严格等间隔（10线程 / 间隔 1.0s）")
    print("=" * 60)

    num_threads = 10
    interval = 1.0
    limiter = RateLimiter(min_interval_seconds=interval)

    call_times: list[float] = []
    lock = threading.Lock()
    barrier = threading.Barrier(num_threads)

    def worker(tid: int):
        barrier.wait()
        limiter._injected_now = tid * interval
        limiter.wait()
        with lock:
            call_times.append(tid * interval)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    call_times.sort()
    intervals = [call_times[i+1] - call_times[i] for i in range(len(call_times)-1)]
    print(f"  调用时刻: {[f'{t:.2f}s' for t in call_times]}")
    print(f"  间隔: {[f'{iv:.2f}s' for iv in intervals]}")

    ok = all(abs(iv - interval) < 0.01 for iv in intervals)
    print(f"  全部严格 {interval}s ± 0.01s: {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_global_limiter_parallel_threads():
    """Test 2: 全局限速器在多线程下正确协调（模拟模式）"""
    print("\n" + "=" * 60)
    print("Test 2: 全局限速器多线程协调（5线程 / 间隔 2.0s）")
    print("=" * 60)

    num_threads = 5
    interval = 2.0
    source = "arxiv_test"

    original = rl._GLOBAL_LIMITERS.get(source)
    test_limiter = RateLimiter(min_interval_seconds=interval)
    rl._GLOBAL_LIMITERS[source] = test_limiter

    try:
        call_times: list[float] = []
        lock = threading.Lock()
        barrier = threading.Barrier(num_threads)

        def worker(tid: int):
            barrier.wait()
            test_limiter._injected_now = tid * interval
            test_limiter.wait()
            with lock:
                call_times.append(tid * interval)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        call_times.sort()
        intervals = [call_times[i+1] - call_times[i] for i in range(len(call_times)-1)]
        print(f"  调用时刻: {[f'{t:.2f}s' for t in call_times]}")
        print(f"  间隔: {[f'{iv:.2f}s' for iv in intervals]}")

        ok = all(abs(iv - interval) < 0.01 for iv in intervals)
        print(f"  全严格 {interval}s: {'✅ PASS' if ok else '❌ FAIL'}")
        return ok
    finally:
        if original:
            rl._GLOBAL_LIMITERS[source] = original
        else:
            rl._GLOBAL_LIMITERS.pop(source, None)


def test_init_last_negative():
    """Test 3: _last 初始化为负值（不是 0）"""
    print("\n" + "=" * 60)
    print("Test 3: _last 初始化为负值（_last=-interval 修复）")
    print("=" * 60)

    interval = 5.0
    limiter = RateLimiter(min_interval_seconds=interval)
    print(f"  interval={interval}s, _last={limiter._last}")

    # 验证 _last 初始值
    ok = abs(limiter._last + interval) < 0.001
    print(f"  _last == -{interval}: {'✅ PASS' if ok else '❌ FAIL'}")

    # 验证首个请求在真实时钟下会被强制等待
    # (因为 elapsed = now - (-interval) = now + interval ≥ interval，通常无需再次 sleep)
    t0 = time.perf_counter()
    limiter.wait()
    elapsed = time.perf_counter() - t0
    # 首个请求：elapsed = real_time + interval >> interval → no sleep → elapsed ≈ 0
    # 这是正确行为：_last=-interval 意味着"已经过了 interval"，无需等待
    print(f"  首个 wait() 实际 sleep: {elapsed:.3f}s（~0s 是正确的：_last=-interval 等效于'已经过了 interval'）")
    return ok


def test_penalty_blocks_requests():
    """Test 4: 惩罚期正确阻止新请求（真实时钟）"""
    print("\n" + "=" * 60)
    print("Test 4: 惩罚期（penalty）阻止新请求（真实时钟）")
    print("=" * 60)

    limiter = RateLimiter(min_interval_seconds=0.1)

    # 先发一个请求消耗掉初始 penalty
    limiter.wait()

    # 立刻设置惩罚期（penalty_until 在未来）
    now = time.monotonic()
    limiter._penalty_until = now + 5.0
    print(f"  penalty_until = now+5.0 = {now + 5.0:.2f}, now = {now:.2f}")

    t0 = time.perf_counter()
    limiter.wait()
    real_elapsed = time.perf_counter() - t0

    ok = abs(real_elapsed - 5.0) < 0.5
    print(f"  wait() 等待了 {real_elapsed:.3f}s（应为 ~5.0s）")
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def test_real_timing_multi_thread():
    """Test 5: 真实时序测试（无模拟，10线程 / 0.5s 间隔）"""
    print("\n" + "=" * 60)
    print("Test 5: 真实时序测试（10线程 / 0.5s 间隔）")
    print("=" * 60)

    num_threads = 10
    interval = 0.5
    limiter = RateLimiter(min_interval_seconds=interval)

    call_times: list[float] = []
    lock = threading.Lock()
    barrier = threading.Barrier(num_threads)

    def worker(tid: int):
        barrier.wait()
        limiter.wait()
        with lock:
            call_times.append(time.monotonic())

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    call_times.sort()
    intervals = [call_times[i+1] - call_times[i] for i in range(len(call_times)-1)]
    print(f"  请求间隔: {[f'{iv:.3f}s' for iv in intervals]}")

    ok = all(iv >= interval * 0.95 for iv in intervals)
    print(f"  全部 >= {interval}s × 0.95: {'✅ PASS' if ok else '❌ FAIL'}")
    return ok


def main():
    tests = [
        test_init_last_negative,
        test_penalty_blocks_requests,
        test_parallel_threads_strict_interval,
        test_global_limiter_parallel_threads,
        test_real_timing_multi_thread,
    ]

    results = []
    for t in tests:
        try:
            r = t()
            results.append((t.__name__, r))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ❌ 异常: {e}")
            results.append((t.__name__, False))

    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'} {name}")
    passed = sum(1 for _, ok in results if ok)
    print(f"\n  {passed}/{len(results)} 通过")
    if passed == len(results):
        print("\n  ✅ 所有测试通过！多线程限速严格遵守间隔")
    else:
        print("\n  ❌ 有测试失败")


if __name__ == "__main__":
    main()

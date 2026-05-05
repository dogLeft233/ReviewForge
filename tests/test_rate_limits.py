"""验证 rate_limits 配置和 adapter 初始化（无网络调用）"""
import asyncio
from src.executor import RetrieverManager
from src.rate_limits import SOURCE_RATE_LIMITS, get_source_config


def test_source_configs():
    """验证每个 source 的配置"""
    print("=== Source Configs ===")
    expected_sources = {
        "arxiv", "semantic_scholar", "dblp",
        "github", "huggingface", "hackernews", "serper",
    }
    actual_sources = set(SOURCE_RATE_LIMITS.keys())
    assert expected_sources == actual_sources, \
        f"Mismatch: expected {expected_sources}, got {actual_sources}"

    for name, cfg in SOURCE_RATE_LIMITS.items():
        print(f"  {name:20s}: min_interval={cfg.min_interval_seconds:.1f}s  "
              f"burst={cfg.burst}  no_key_conc={cfg.no_key_concurrency}  "
              f"key_conc={cfg.key_concurrency}  "
              f"{'REQUIRES KEY' if cfg.requires_key else ''}  {cfg.note[:40]}")
        # 验证基础约束
        assert cfg.no_key_concurrency >= 0, f"{name}: no_key_concurrency must be >= 0"
        assert cfg.key_concurrency >= cfg.no_key_concurrency, \
            f"{name}: key_concurrency must be >= no_key_concurrency"

    print("\n✅ test_source_configs passed")


def test_adapter_config():
    """验证 AsyncRetrieverAdapter 使用了正确的并发数和间隔"""
    print("\n=== Adapter Configs ===")
    with RetrieverManager() as mgr:
        from src.executor import AsyncRetrieverAdapter

        for name, adapter in mgr._retrievers.items():
            if not isinstance(adapter, AsyncRetrieverAdapter):
                continue
            cfg = get_source_config(name)
            actual_conc = adapter._semaphore._value
            actual_interval = adapter._min_interval

            print(f"  {name:20s}: semaphore={actual_conc}  "
                  f"min_interval={actual_interval:.1f}s  "
                  f"(expected conc={cfg.no_key_concurrency}, "
                  f"interval={cfg.min_interval_seconds:.1f}s)")

            assert actual_conc == cfg.no_key_concurrency, \
                f"{name}: expected {cfg.no_key_concurrency}, got {actual_conc}"
            assert actual_interval == cfg.min_interval_seconds, \
                f"{name}: expected {cfg.min_interval_seconds}, got {actual_interval}"
            assert adapter._last_called == 0.0, \
                f"{name}: _last_called should start at 0.0"

    print("\n✅ test_adapter_config passed")


async def test_rate_limit_sleep():
    """验证 _enforce_rate_limit 真的会 sleep"""
    import time

    print("\n=== Rate Limit Sleep ===")

    with RetrieverManager() as mgr:
        adapter = mgr._retrievers["arxiv"]
        # arxiv: min_interval = 3.0s

        # Manually set _last_called to now to force a sleep
        loop = asyncio.get_running_loop()
        adapter._last_called = loop.time() - 2.0  # 2 seconds ago

        t0 = time.monotonic()
        await adapter._enforce_rate_limit()
        elapsed = time.monotonic() - t0

        print(f"  elapsed={elapsed:.2f}s (should be ~1.0s, since we needed 1.0s more of 3.0s)")
        assert 0.9 <= elapsed <= 1.5, f"Expected ~1.0s sleep, got {elapsed:.2f}s"

        # Now check that _last_called was updated
        assert adapter._last_called > loop.time() - 0.1, \
            "_last_called should be updated to near current time"

    print("\n✅ test_rate_limit_sleep passed")


async def test_no_sleep_when_no_limit():
    """验证 min_interval=0 时不 sleep"""
    import time

    print("\n=== No Sleep When No Limit ===")

    with RetrieverManager() as mgr:
        # github has min_interval = 0.0
        adapter = mgr._retrievers["github"]
        adapter._last_called = asyncio.get_running_loop().time() - 10.0

        t0 = time.monotonic()
        await adapter._enforce_rate_limit()
        elapsed = time.monotonic() - t0

        print(f"  elapsed={elapsed:.4f}s (should be ~0s)")
        assert elapsed < 0.05, f"Should not sleep, but took {elapsed:.4f}s"

    print("\n✅ test_no_sleep_when_no_limit passed")


def main():
    test_source_configs()
    test_adapter_config()
    asyncio.run(test_rate_limit_sleep())
    asyncio.run(test_no_sleep_when_no_limit())
    print("\n🎉 All tests passed")


if __name__ == "__main__":
    main()

from relay_d.utils.tf_filter import StaleTfFilter

NS = 10**9


def test_old_stamps_are_dropped_and_summarized_once(monkeypatch):
    now = [0.0]
    warnings = []
    monkeypatch.setattr("relay_d.utils.tf_filter.logger.warning", warnings.append)
    f = StaleTfFilter(cache_time=10.0, warn_every=10.0, clock=lambda: now[0])
    assert f.accept("link", 1000 * NS)
    assert f.accept("link", 995 * NS)  # inside the 10 s cache: tf2 keeps it
    assert f.accept("other", 5 * NS)  # a frame has its own newest stamp
    for _ in range(50):  # a replayed bag: stamps far in the past, over and over
        assert not f.accept("link", 100 * NS)
    assert warnings == []  # first summary only after warn_every
    now[0] = 11.0
    assert not f.accept("link", 100 * NS)
    assert len(warnings) == 1 and "link" in warnings[0] and "900 s old" in warnings[0]
    assert not f.accept("link", 100 * NS) and len(warnings) == 1  # rate limited
    assert f.accept("link", 1001 * NS)  # fresh data still passes

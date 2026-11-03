"""The Qt-free settings store: defaults, validation, migration, atomic save."""

import json

import pytest

from relay_d.utils import user_settings as us


def test_table_is_consistent():
    keys = [s.key for s in us.SETTINGS]
    assert len(keys) == len(set(keys))
    assert {s.group for s in us.SETTINGS} <= set(us.GROUPS)
    for s in us.SETTINGS:
        assert us._valid(s, s.default), s.key  # every default is valid for its own type
        if s.kind == "choice":
            assert s.default in s.choices, s.key
        if s.minimum is not None and s.kind in ("int", "float"):
            assert s.minimum <= s.default <= s.maximum, s.key


def test_defaults_without_a_file(settings_file):
    assert not settings_file.exists()
    assert us.get("rec_default_frequency") == 30 and us.get("theme") == "light"
    assert us.values() == {**us.defaults()}


def test_save_merges_keeps_unknown_keys_and_leaves_no_temp_file(settings_file):
    settings_file.write_text(json.dumps({"some_future_key": 1, "theme": "dark"}))
    us.reload()
    result = us.save({"rec_default_frequency": 60})
    on_disk = json.loads(settings_file.read_text())
    assert on_disk == {"some_future_key": 1, "theme": "dark", "rec_default_frequency": 60}
    assert result["rec_default_frequency"] == 60 and us.get("theme") == "dark"
    assert [p.name for p in settings_file.parent.iterdir()] == ["settings.json"]


def test_invalid_values_are_rejected_on_save_and_ignored_on_load(settings_file):
    with pytest.raises(ValueError):
        us.save({"rec_default_frequency": "fast"})
    with pytest.raises(ValueError):
        us.save({"theme": "purple"})
    with pytest.raises(ValueError):
        us.save({"confirm_destructive": 1})  # an int is not a bool
    assert not settings_file.exists()
    settings_file.write_text(json.dumps({"rec_default_frequency": "x", "theme": "dark"}))
    us.reload()
    assert us.get("rec_default_frequency") == 30 and us.get("theme") == "dark"


def test_corrupt_or_wrong_file_means_defaults(settings_file):
    settings_file.write_text("{ not json")
    assert us.reload()["theme"] == "light"
    settings_file.write_text("[1, 2]")
    assert us.reload()["theme"] == "light"


def test_old_keys_are_migrated(settings_file):
    settings_file.write_text(json.dumps({"update_rate": 12, "enable_compression": True}))
    assert us.reload()["visualizer_update_rate"] == 12


def test_float_settings_accept_ints(settings_file):
    us.save({"rec_warmup_s": 3})
    assert us.get("rec_warmup_s") == 3.0 and isinstance(us.get("rec_warmup_s"), float)


def test_restart_keys_and_compression_kwargs(settings_file):
    old = us.values()
    new = {**old, "ros_domain_id": 5, "theme": "dark"}
    assert us.changed_restart_keys(old, new) == ["ros_domain_id"]  # the theme changes live
    assert us.h5_compression() == {"compression": "gzip", "compression_opts": 4}
    us.save({"hdf5_compression": "lzf"})
    assert us.h5_compression() == {"compression": "lzf"}
    us.save({"hdf5_compression": "none"})
    assert us.h5_compression() == {}


def test_values_equal_to_the_default_are_not_stored(settings_file):
    us.save({"rec_default_frequency": 60, "theme": "light", "infer_hz": 10.0})
    assert json.loads(settings_file.read_text()) == {"rec_default_frequency": 60}
    us.save({"rec_default_frequency": 30})  # back to the default: the key disappears
    assert json.loads(settings_file.read_text()) == {}
    assert us.get("rec_default_frequency") == 30


def test_resetting_a_migrated_setting_does_not_bring_the_old_value_back(settings_file):
    settings_file.write_text(json.dumps({"update_rate": 12}))
    assert us.reload()["visualizer_update_rate"] == 12
    us.save({"visualizer_update_rate": 20})
    assert us.get("visualizer_update_rate") == 20 and "update_rate" not in json.loads(settings_file.read_text())

"""Start-up and live-change decisions of the app (the window itself needs ROS, these functions do not)."""

from relay_d.acquisition.qt_app.settings_apply import live_changes, ros_domain_to_set, startup_page


def test_ros_domain_id_only_when_set_and_not_in_the_environment():
    assert ros_domain_to_set({"ros_domain_id": 7}, {}) == "7"
    assert ros_domain_to_set({"ros_domain_id": 0}, {}) == "0"  # 0 is a real domain
    assert ros_domain_to_set({"ros_domain_id": -1}, {}) is None  # -1: leave the environment alone
    assert ros_domain_to_set({"ros_domain_id": 7}, {"ROS_DOMAIN_ID": "3"}) is None  # the environment wins


def test_startup_page_falls_back_to_help():
    assert startup_page({"startup_page": "review"}) == "review"
    assert startup_page({"startup_page": "help"}) == "help"
    assert startup_page({"startup_page": "nonsense"}) == "help"
    assert startup_page({}) == "help"


def test_live_changes_lists_only_what_differs():
    assert live_changes({"theme": "dark", "ros_spin_ms": 10}, "light", 10) == {"theme": "dark"}
    assert live_changes({"theme": "light", "ros_spin_ms": 25}, "light", 10) == {"spin_ms": 25}
    assert live_changes({"theme": "light", "ros_spin_ms": 10}, "light", 10) == {}
    assert live_changes({}, "light", 10) == {}


def test_only_the_record_page_needs_a_config():
    from relay_d.acquisition.qt_app.settings_apply import needs_config

    assert needs_config("start", False) and not needs_config("start", True)
    assert not any(needs_config(p, False) for p in ("data", "import", "review", "train", "settings", "help"))

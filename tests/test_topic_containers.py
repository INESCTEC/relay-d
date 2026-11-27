"""Container cards show their input name whatever the input is called (topic_type is the name's first word)."""

import pytest


@pytest.mark.parametrize("input_name, label", [
    ("camera_rgb", "label_topic_name"),  # "camera" has no .ui of its own: imageWidget.ui
    ("image_1", "label_topic_name"),
    ("joint_1", "text_data_display"),
])
def test_card_shows_input_name_and_status_off_the_badges(qapp, input_name, label):
    from PyQt5.QtWidgets import QWidget
    from relay_d.acquisition.qt_app.ui.containers.topic_containers import TopicContainers

    tc = TopicContainers(QWidget())  # a bare widget stands in for the main window
    container = tc.create_container(input_name.split("_")[0], {"input_name": input_name})
    tc.update_container_info(container, {"input_name": input_name, "topic_path": "/t", "status": "Ready (press Start Recording)"})

    assert getattr(container, label).text() == input_name
    assert getattr(container, label).toolTip() == "/t"
    for badge in ("label_frame_rate", "label_timestamp", "positionLabel", "velocityLabel", "torqueLabel"):
        if hasattr(container, badge):
            assert "Ready" not in getattr(container, badge).text()

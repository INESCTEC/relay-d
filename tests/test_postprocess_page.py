"""Post-process page: pre-check table, output preview, Convert state and the recordings row."""

import os
import time

import pytest

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QMainWindow  # noqa: E402

from relay_d.acquisition.qt_app.ui.config.yaml_parser import YamlParser  # noqa: E402
from relay_d.acquisition.qt_app.ui.postprocess import postprocess_page as pp  # noqa: E402


def _report(ok=(True, False), error=None):
    demos = [
        {"file": "/rec/a.h5", "demo": f"demo_{i}", "samples": 50 + i, "rate_hz": 20.0,
         "recorded_with": "config_peter.yaml", "recorded_hz": 20.0, "ok": good,
         "reason": "" if good else "no robot_gripper_data (action)"}
        for i, good in enumerate(ok)
    ]
    converted = sum(ok)
    return {"format": "robomimic", "error": error, "demos": [] if error else demos,
            "output": [] if error else [["data/demo_0/actions", "(N,8)"], ["mask/train", f"{converted} demos"]],
            "converted": 0 if error else converted, "total": len(ok)}


@pytest.fixture
def page(monkeypatch, qapp):
    import relay_d.acquisition.qt_app as qt_app

    monkeypatch.syspath_prepend(os.path.dirname(qt_app.__file__))
    from relay_d.acquisition.qt_app.ui.containers.mainWindow_ui import Ui_MainWindow

    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)
    p = pp.PostProcessPage(ui, YamlParser())
    p._window = window
    p.yaml_converter.data_config = {"output": {"streams": {"x": {}}}, "actions_to_extract": ["x"]}
    p.output_format = "robomimic"
    return p


def _wait(qapp, cond, timeout=5.0):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        qapp.processEvents()
        time.sleep(0.01)
    assert cond()


def test_good_report_fills_table_and_enables_convert(page, qapp, monkeypatch):
    monkeypatch.setattr(page.yaml_converter, "precheck", lambda fmt, files: _report(), raising=False)
    page.set_files(["/rec/a.h5"])
    _wait(qapp, lambda: page._report is not None)  # the real background runner
    t = page.table_precheck
    assert t.rowCount() == 2
    assert [t.item(0, c).text() for c in range(4)] == ["demo_0", "50", "20.0", "config_peter.yaml (20 Hz)"]
    assert t.item(0, 4).text().startswith("✔") and "robot_gripper_data" in t.item(1, 4).text()
    assert page.btn_convert.isEnabled() and page.btn_convert.text() == "Convert 1 demo"
    assert page.label_summary.text() == "1 of 2 demos will be converted"
    assert "data/demo_0/actions" in page.txt_output.toPlainText()
    assert page.label_recordings.text() == "2 demos in 1 file"
    assert page.label_precheck_error.isHidden()


@pytest.mark.parametrize("report, reason", [
    (_report(error="robot0_gripper is in none of the recordings"), "robot0_gripper is in none"),
    (_report(ok=(False, False)), "No demo would be converted"),
])
def test_convert_disabled_with_reason(page, qapp, monkeypatch, report, reason):
    monkeypatch.setattr(page.yaml_converter, "precheck", lambda fmt, files: report, raising=False)
    page.set_files(["/rec/a.h5"])
    _wait(qapp, lambda: page._report is not None)
    assert not page.btn_convert.isEnabled()
    assert reason in page.btn_convert.toolTip()
    if report["error"]:  # shown once, in the error box; the side label points to it
        assert not page.label_precheck_error.isHidden() and reason in page.label_precheck_error.text()
        assert page.label_convert_reason.text() == "See the error above."
    else:
        assert reason in page.label_convert_reason.text()


def test_nothing_to_check_disables_convert_without_running(page, monkeypatch):
    monkeypatch.setattr(pp, "run_in_background", lambda *a: pytest.fail("no pre-check without files"))
    page.set_files([])
    assert not page.btn_convert.isEnabled() and "Add recordings" in page.btn_convert.toolTip()
    page._select_output_format("robomimic")  # deselect
    assert "output format" in page.label_convert_reason.text()


def test_add_folder_collects_recursively(page, tmp_path, monkeypatch):
    (tmp_path / "s1" / "deep").mkdir(parents=True)
    for rel in ("top.h5", "s1/a.h5", "s1/deep/b.h5", "s1/notes.txt"):
        (tmp_path / rel).write_text("")
    monkeypatch.setattr(pp, "run_in_background", lambda *a: None)
    monkeypatch.setattr(pp.QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
    page._select_folder_to_convert()
    assert page.h5_files == pp.collect_inputs([str(tmp_path)])
    assert sorted(os.path.relpath(f, tmp_path) for f in page.h5_files) == ["s1/a.h5", "s1/deep/b.h5", "top.h5"]
    page.btn_clear_files.click()
    assert page.h5_files == []


def test_later_refresh_wins(page, monkeypatch):
    jobs = []
    monkeypatch.setattr(pp, "run_in_background", lambda parent, work, on_done: jobs.append(on_done))
    page.set_files(["/rec/a.h5"])
    page.set_files(["/rec/a.h5", "/rec/b.h5"])
    first, second = jobs
    second(_report(ok=(True, True)), None)
    first(_report(ok=(False,)), None)  # finishes last but was started first: ignored
    assert page.table_precheck.rowCount() == 2 and page.btn_convert.text() == "Convert 2 demos"


def test_selected_recordings_can_be_removed_before_converting(page, qapp, monkeypatch):
    from PyQt5.QtCore import QItemSelectionModel, Qt
    from PyQt5.QtTest import QTest

    files = ["/rec/a.h5", "/rec/b.h5", "/rec/c.h5"]
    checked = []

    def precheck(fmt, chosen):
        checked.append(list(chosen))
        demos = [{"file": f, "demo": os.path.basename(f), "samples": 50, "rate_hz": 20.0, "recorded_with": None,
                  "recorded_hz": None, "ok": True, "reason": ""} for f in chosen]
        return {"format": fmt, "error": None, "demos": demos, "output": [], "converted": len(demos), "total": len(demos)}

    monkeypatch.setattr(page.yaml_converter, "precheck", precheck, raising=False)
    page.set_files(files)
    _wait(qapp, lambda: page._report is not None and page.table_precheck.rowCount() == 3)
    assert not page.btn_remove_files.isEnabled()  # nothing selected yet

    table = page.table_precheck
    for row in (0, 2):
        table.selectionModel().select(table.model().index(row, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)
    assert page.btn_remove_files.isEnabled()
    page.btn_remove_files.click()
    assert page.h5_files == ["/rec/b.h5"]  # only left out of the selection: nothing is deleted from disk
    _wait(qapp, lambda: checked[-1] == ["/rec/b.h5"] and table.rowCount() == 1)

    table.selectRow(0)
    table.setFocus()
    QTest.keyClick(table, Qt.Key_Delete)
    assert page.h5_files == []

import pytest

pytest.importorskip("PyQt5")
import os

from relay_d.acquisition.qt_app.utils import default_configs as dc


def test_generate_copies_templates_and_remembers_folder(tmp_path, monkeypatch):
    saved = {}
    monkeypatch.setattr(dc.QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(dc.config, "save_settings", saved.update)
    monkeypatch.setattr(dc.CustomMessageBox, "info", lambda *a, **k: None)

    assert dc.generate_default_configs() == str(tmp_path)
    assert sorted(os.listdir(tmp_path)) == sorted(dc.TEMPLATES)
    assert saved == {"config_folder": str(tmp_path)}

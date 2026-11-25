"""YamlParser: a bad file must not half-replace the loaded config; the file text is shown and saved as written."""

from relay_d.acquisition.qt_app.ui.config.yaml_parser import YamlParser

GOOD = "config:\n  dataset_name: a  # keep me\n  input:\n    cam: {topic: /cam}\n  output: {}\n"


def test_empty_sections_load_and_a_bad_file_keeps_the_previous_config(tmp_path):
    a, b, bad = tmp_path / "a.yaml", tmp_path / "b.yaml", tmp_path / "bad.yaml"
    a.write_text(GOOD)
    b.write_text("config:\n  dataset_name: b\n  input:\n  output:\n")  # empty sections are None, not {}
    bad.write_text("config: [not, a, mapping]\n")
    p = YamlParser()
    assert p.load_yaml_file(str(b)) and p.get_input_data() == [] and p.get_streams() == {} and p.get_output_bridge_map() == {}
    assert p.load_yaml_file(str(a)) and p.get_dataset_name() == "a"
    assert not p.load_yaml_file(str(bad))
    assert p.get_current_config_path() == str(a) and p.yaml_data["config"]["dataset_name"] == "a"
    assert [i["name"] for i in p.get_input_data()] == ["cam"]


def test_preview_and_save_keep_comments_and_key_order(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(GOOD)
    p = YamlParser()
    assert p.load_yaml_file(str(path))
    assert "# keep me" in p.convert_yaml_to_html()
    edited = GOOD.replace("dataset_name: a", "dataset_name: z")
    assert p.load_from_yaml_string(edited) and p.save_yaml_file(data=edited)
    assert path.read_text() == edited and not (tmp_path / "c.yaml.tmp").exists()

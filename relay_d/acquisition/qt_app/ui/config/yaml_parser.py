import os
import yaml
import re
import html

from relay_d.utils.coloring_logger import logger


# VSCode Dark+ theme colors for YAML
VSCODE = {
    "key":      "#1A7DC4",   # dark blue    — keys
    "string":   "#A0522D",   # dark sienna  — string values
    "number":   "#4A7C3F",   # dark green   — numbers
    "boolean":  "#1C5FA8",   # dark blue    — true / false
    "null":     "#1C5FA8",   # dark blue    — null / ~
    "comment":  "#3D6B35",   # dark green   — comments
    "dash":     "#555555",   # dark grey    — list dash (-)
    "colon":    "#555555",   # dark grey    — colons
    "anchor":   "#7B3A9E",   # dark purple  — anchors & aliases
}


class YamlParser:
    def __init__(self):
        logger.info("Create YamlParser object.")

        self.yaml_data = None
        self.yaml_text = None  # the text yaml_data was loaded from, shown as is (comments, key order)
        self.current_config = {}
        self.clear()

    def load_yaml_file(self, file_path):
        try:
            with open(file_path, "r") as stream:
                self._load(stream.read())
            self.current_config = file_path
            self._print_parsed_data()
            return True
        except Exception as exc:
            logger.error(f"Error loading YAML file: {exc}")
            return False

    def _load(self, text):
        """Parse ``text`` and only then replace the loaded data: a bad file leaves the previous config intact."""
        data = yaml.safe_load(text)
        parsed = self._retrieve_all_parameters(data)
        self.yaml_data, self.yaml_text = data, text
        self.clear()
        for name, value in parsed.items():
            setattr(self, name, value)

    def _retrieve_all_parameters(self, data):
        logger.info("Retrieving YAML parameters")

        parsed = {}
        if not data or "config" not in data:
            logger.error("Invalid YAML: missing 'config'")
            return parsed

        config = data["config"]
        parsed["dataset_name"] = config.get("dataset_name", "demo")

        if "input" in config:
            # Also add a raw for future procesing
            input_section = config.get("input") or {}
            parsed["yaml_input_data_raw"] = input_section
            parsed["yaml_input_data"] = inputs = []

            for name, input_config in input_section.items():
                if not isinstance(input_config, dict):
                    continue

                input_data = {"name": name}

                has_parent = "parent_frame" in input_config
                has_child = "child_frame" in input_config

                if has_parent and has_child:
                    input_data["is_tf"] = True
                    input_data["parent_frame"] = input_config["parent_frame"]
                    input_data["child_frame"] = input_config["child_frame"]
                    input_data["tf_topic"] = input_config.get("tf_topic", "/tf")
                    input_data["tf_static_topic"] = input_config.get(
                        "tf_static_topic", "/tf_static"
                    )
                    input_data["tf_wait_timeout"] = input_config.get(
                        "tf_wait_timeout", None
                    )
                    input_data["topic"] = None
                    input_data["fill_gaps"] = input_config.get("fill_gaps", False)
                elif "topic" in input_config:
                    input_data["topic"] = input_config["topic"]
                    input_data["is_tf"] = False
                    input_data["joint_names"] = input_config.get("joint_names", [])
                    input_data["fill_gaps"] = input_config.get("fill_gaps", False)
                else:
                    continue

                inputs.append(input_data)

        if "metadata" in config:
            parsed["metadata"] = config["metadata"]

        output_section = config.get("output") or {}
        if isinstance(output_section.get("streams"), dict):
            parsed["yaml_streams_data"] = output_section["streams"]
        return parsed

    def _print_parsed_data(self):
        """Print all parsed data for debugging. Lines with nothing meaningful
        to show are skipped rather than printed blank."""
        logger.info("\n*** INPUT DATA ***")
        for item in self.yaml_input_data:
            if item.get("is_tf"):
                logger.info(
                    f"  {item['name']}: TF {item['parent_frame']} -> "
                    f"{item['child_frame']} (tf_topic={item['tf_topic']})"
                )
            elif item.get("topic"):
                logger.info(f"  {item['name']}: {item['topic']}")

        logger.info("\n*** OUTPUT DATA ***")
        for key, path in self.get_output_bridge_map().items():
            logger.info(f"  {key}: {path}")
        for name in self.yaml_streams_data:
            logger.info(f"  stream {name}")


    # ------------------------------------------------------------------
    # HTML / syntax highlighting
    # ------------------------------------------------------------------

    def _span(self, color: str, text: str) -> str:
        """Wrap *already-escaped* text in a coloured span."""
        return f'<span style="color:{color}">{text}</span>'

    def highlight_yaml_syntax_line(self, line: str) -> str:
        """
        Apply VSCode Dark+-style colouring to a single raw YAML line.
        Returns an HTML-safe string.
        """
        # Work on the escaped version from the start so we never double-escape.
        escaped = html.escape(line)

        # --- comment lines (must come first, they swallow the rest) ---
        comment_match = re.match(r'^(\s*)(#.*)$', escaped)
        if comment_match:
            indent, comment = comment_match.group(1), comment_match.group(2)
            return indent + self._span(VSCODE["comment"], comment)

        # Split leading indent + optional list dash from the rest
        prefix_match = re.match(r'^(\s*)(- ?)?(.*)$', escaped)
        indent = prefix_match.group(1) if prefix_match else ""
        dash   = prefix_match.group(2) if prefix_match and prefix_match.group(2) else ""
        rest   = prefix_match.group(3) if prefix_match else escaped.lstrip()

        result = indent
        if dash:
            result += self._span(VSCODE["dash"], dash)

        # key: value  (inline comment optional)
        kv_match = re.match(
            r'^([a-zA-Z_][a-zA-Z0-9_ -]*)(\s*:\s*)(.*?)(\s*#.*)?$', rest
        )
        if kv_match:
            key     = kv_match.group(1)
            colon   = kv_match.group(2)
            value   = kv_match.group(3)
            comment = kv_match.group(4) or ""

            result += self._span(VSCODE["key"], key)
            result += self._span(VSCODE["colon"], colon)
            result += self._colour_value(value)
            if comment:
                result += self._span(VSCODE["comment"], comment)
            return result

        # bare list item value (no key)
        if rest:
            result += self._colour_value(rest)

        return result

    def _colour_value(self, value: str) -> str:
        """Return a coloured span for a YAML scalar value (already html-escaped)."""
        v = value.strip()

        if not v:
            return value

        # boolean
        if re.fullmatch(r'true|false|True|False|TRUE|FALSE', v):
            return value.replace(v, self._span(VSCODE["boolean"], v), 1)

        # null
        if re.fullmatch(r'null|Null|NULL|~', v):
            return value.replace(v, self._span(VSCODE["null"], v), 1)

        # number (int or float)
        if re.fullmatch(r'-?\d+(?:\.\d+)?', v):
            return value.replace(v, self._span(VSCODE["number"], v), 1)

        # quoted string
        if (v.startswith('"') and v.endswith('"')) or \
           (v.startswith("'") and v.endswith("'")):
            return value.replace(v, self._span(VSCODE["string"], v), 1)

        # unquoted string (everything else with a non-empty value)
        return value.replace(v, self._span(VSCODE["string"], v), 1)

    def convert_yaml_to_html(self) -> str:
        """
        Convert loaded YAML data to an HTML <pre> block with VSCode Dark+ colours.
        Every line is highlighted individually via highlight_yaml_syntax_line().
        """
        if not self.yaml_data:
            return "<pre></pre>"

        if self.yaml_text is not None and yaml.safe_load(self.yaml_text) == self.yaml_data:
            raw = self.yaml_text  # unchanged since it was loaded: keep its comments and key order
        else:
            raw = yaml.dump(self.yaml_data, default_flow_style=False, allow_unicode=True)
            if raw.startswith("---\n"):
                raw = raw[4:]

        highlighted_lines = [
            self.highlight_yaml_syntax_line(line) for line in raw.splitlines()
        ]
        body = "<br>".join(highlighted_lines)

        return (
            '<pre style=\'color:#333333; padding:12px; '
            'border-radius:6px; font-family:"JetBrains Mono","Cascadia Code",'
            '"Fira Code",monospace; font-size:12px; line-height:1.5;\'>'
            f"{body}</pre>"
        )

    # ------------------------------------------------------------------
    # File I/O helpers (unchanged)
    # ------------------------------------------------------------------

    def save_yaml_file(self, file_path=None, data=None):
        """Save the current YAML data (or ``data``: a dict is dumped, a string is written as is) to a file, atomically."""
        if file_path is None:
            file_path = self.current_config
        if not file_path:
            logger.warning("No file path specified for saving")
            return False

        if data is None:
            data = self.yaml_data
        tmp = f"{file_path}.tmp"
        try:
            with open(tmp, "w") as f:
                if isinstance(data, str):
                    f.write(data)
                else:
                    yaml.dump(data, f, default_flow_style=False)
            os.replace(tmp, file_path)
            logger.info(f"YAML file saved successfully: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving YAML file: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)
            return False

    def load_from_yaml_string(self, yaml_string):
        """Load YAML data from a string (used for editing)"""
        try:
            self._load(yaml_string)
            return True
        except Exception as e:
            logger.error(f"Error parsing YAML string: {e}")
            return False

    # ------------------------------------------------------------------
    # Getters (unchanged)
    # ------------------------------------------------------------------

    def is_loaded(self):
        return bool(self.current_config)

    def clear(self):
        """Reset the parsed lists. ``yaml_data`` and ``current_config`` are kept: this also
        runs at the start of every load. To unload a config, reset those two as well."""
        self.yaml_input_data = []
        self.yaml_input_data_raw = {}
        self.yaml_streams_data = {}
        self.dataset_name = "demo"
        self.metadata = {}

    def get_current_config_path(self):
        return self.current_config

    def get_input_data(self):
        return self.yaml_input_data

    def get_input_data_raw(self):
        return self.yaml_input_data_raw

    def get_dataset_name(self):
        return self.dataset_name

    def get_metadata(self):
        return self.metadata

    def get_streams(self) -> dict:
        return self.yaml_streams_data

    def get_output_bridge_map(self) -> dict:
        """Flat {logical_key: recorded_path} bridge-map entries from
        `config.output` — top-level keys other than `streams`/`prefix_path`,
        e.g. `agentview_image: camera_rgb/images`. Mirrors the converter's
        (yaml_driven_converter.DataTransformEngine._build_bridge_map).
        """
        if not self.yaml_data:
            return {}
        output_section = (self.yaml_data.get("config") or {}).get("output") or {}
        return {
            key: value
            for key, value in output_section.items()
            if key not in ("streams", "prefix_path") and isinstance(value, str)
        }

    def get_obs_config(self):
        # Bridge-map entries (camera/organized-pointcloud obs, e.g.
        # `agentview_image: camera_rgb/images`) become passthrough obs in
        # obs_config.yaml. The recorded group is always the input name, so the
        # part before the first '/' is the input ObsBuilder decodes live.
        bridge_obs = {
            key: {"source": path.split("/")[0]}
            for key, path in self.get_output_bridge_map().items()
        }
        obs_config = {
            "input": self.get_input_data_raw(),
            "observations": {**self.get_streams(), **bridge_obs},
        }
        # Omitting `sync` disables the timestamp-gap check (see obs_config_template.yaml),
        # so only emit it when the acquisition config sets sync_tolerance_sec.
        threshold = self.yaml_data["config"].get("sync_tolerance_sec")
        if threshold is not None:
            obs_config = {"sync": {"threshold": threshold}, **obs_config}
        return {"config": obs_config}
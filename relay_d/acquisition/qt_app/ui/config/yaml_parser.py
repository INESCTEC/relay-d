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
        self.yaml_config_data = []
        self.yaml_input_data = []
        self.yaml_output_data = []
        self.yaml_topic_data = []
        self.yaml_streams_data = {}
        self.yaml_actions_to_extract = []
        self.current_config = {}
        self.prefix_path = ""
        self.dataset_name = "demo"
        self.metadata = {}
        self.sync_tolerance_sec = None
        self.sync_mode = "soft"

    def load_yaml_file(self, file_path):
        try:
            with open(file_path, "r") as stream:
                self.yaml_data = yaml.safe_load(stream)
                self._retrieve_all_parameters()
                self.current_config = file_path
                self._print_parsed_data()
                return True
        except Exception as exc:
            logger.error(f"Error loading YAML file: {exc}")
            return False

    def _retrieve_all_parameters(self):
        logger.info("Retrieving YAML parameters")

        self.clear()

        if not self.yaml_data or "config" not in self.yaml_data:
            logger.error("Invalid YAML: missing 'config'")
            return

        config = self.yaml_data["config"]
        self.dataset_name = config.get("dataset_name", "demo")

        if "data" in config:
            self.yaml_config_data = [
                {"name": name, "count": count} for name, count in config["data"].items()
            ]

        if "input" in config:
            global_transfer_rate = -1
            if "recording_frequency" in config:
                global_transfer_rate = config["recording_frequency"]
            elif "metadata" in config and "transfer_rate" in config["metadata"]:
                global_transfer_rate = config["metadata"]["transfer_rate"]
            elif "metadata" in config and "recording_frequency" in config["metadata"]:
                global_transfer_rate = config["metadata"]["recording_frequency"]
            else:
                global_transfer_rate = 30

            # Also add a raw for future procesing
            self.yaml_input_data_raw = config["input"]

            for name, input_config in config["input"].items():
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
                    input_data["output_map"] = input_config.get("output_map", name)
                    input_data["transfer_rate"] = input_config.get(
                        "transfer_rate", global_transfer_rate
                    )
                    input_data["fill_gaps"] = input_config.get("fill_gaps", False)
                    input_data["fill_default"] = input_config.get("fill_default", None)
                elif "topic" in input_config:
                    input_data["topic"] = input_config["topic"]
                    input_data["is_tf"] = False
                    input_data["output_map"] = input_config.get("output_map", name)
                    input_data["transfer_rate"] = input_config.get(
                        "transfer_rate", global_transfer_rate
                    )
                    input_data["joint_names"] = input_config.get("joint_names", [])
                    input_data["fill_gaps"] = input_config.get("fill_gaps", False)
                    input_data["fill_default"] = input_config.get("fill_default", None)
                else:
                    continue

                self.yaml_input_data.append(input_data)

        if "output" in config:
            for output_name, output_fields in config["output"].items():
                if output_name == "prefix_path":
                    self.prefix_path = output_fields
                    continue
                if isinstance(output_fields, dict):
                    for field_name, field_path in output_fields.items():
                        self.yaml_output_data.append(
                            {
                                "name": output_name,
                                "field": field_name,
                                "source": field_path,
                            }
                        )

        if "metadata" in config:
            self.metadata = config["metadata"]

        output_section = config.get("output", {})
        if isinstance(output_section.get("streams"), dict):
            self.yaml_streams_data = output_section["streams"]
        else:
            self.yaml_streams_data = {}

        self.yaml_actions_to_extract = config.get("actions_to_extract", [])
        self.sync_tolerance_sec = config.get("sync_tolerance_sec", None)
        self.sync_mode = config.get("sync_mode", "soft")

    def _print_parsed_data(self):
        """Print all parsed data for debugging. Lines with nothing meaningful
        to show are skipped rather than printed blank."""
        logger.info("\n*** CONFIG DATA ***")
        for item in self.yaml_config_data:
            logger.info(f"  {item['name']}: {item['count']}")

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
        if self.prefix_path:
            logger.info(f"  Prefix path: {self.prefix_path}")
        for item in self.yaml_output_data:
            if not item.get("source"):
                continue
            parent_info = f" (parent: {item['parent']})" if "parent" in item else ""
            logger.info(f"  {item['name']}.{item['field']}: {item['source']}{parent_info}")

        if self.yaml_topic_data:
            logger.info("\n=== TOPIC TYPES ===")
            for item in self.yaml_topic_data:
                logger.info(f"  {item['name']}: {item['type']}")

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

    def save_yaml_file(self, file_path=None, data:str = None):
        """Save the current YAML data to a file"""
        if file_path is None:
            file_path = self.current_config
        if not file_path:
            logger.warning("No file path specified for saving")
            return False

        try:
            with open(file_path, "w") as f:
                yaml.dump(data if data is not None else self.yaml_data, f, default_flow_style=False)
            logger.info(f"YAML file saved successfully: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving YAML file: {e}")
            return False

    def save_specific_to_yaml(self, file_path=None, content=list):

        return True

    def load_from_yaml_string(self, yaml_string):
        """Load YAML data from a string (used for editing)"""
        try:
            self.yaml_data = yaml.safe_load(yaml_string)
            self._retrieve_all_parameters()
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
        self.yaml_config_data = []
        self.yaml_input_data = []
        self.yaml_output_data = []
        self.yaml_topic_data = []
        self.yaml_streams_data = {}
        self.yaml_actions_to_extract = []
        self.prefix_path = ""
        self.dataset_name = "demo"
        self.metadata = {}
        self.sync_tolerance_sec = None
        self.sync_mode = "soft"

    def get_current_config_path(self):
        return self.current_config

    def get_raw_yaml(self):
        if self.yaml_data:
            return yaml.dump(self.yaml_data, default_flow_style=False)
        return ""

    def get_config_data(self):
        return self.yaml_config_data

    def get_input_data(self):
        return self.yaml_input_data

    def get_input_data_raw(self):
        return self.yaml_input_data_raw

    def get_tf_data(self):
        tf_data = []
        for entry in self.yaml_input_data:
            if entry.get("is_tf", False):
                tf_data.append(
                    {
                        "name": entry["name"],
                        "parent_frame": entry.get("parent_frame"),
                        "child_frame": entry.get("child_frame"),
                        "transfer_rate": entry.get("transfer_rate", -1),
                    }
                )
        return tf_data

    def get_output_data(self):
        return self.yaml_output_data

    def get_topic_data(self):
        return self.yaml_topic_data

    def get_prefix_path(self):
        return self.prefix_path

    def get_dataset_name(self):
        return self.dataset_name

    def get_metadata(self):
        return self.metadata

    def get_streams(self) -> dict:
        return self.yaml_streams_data

    def get_output_bridge_map(self) -> dict:
        """Flat {logical_key: recorded_path} bridge-map entries from
        `config.output` — top-level keys other than `streams`/`prefix_path`,
        e.g. `agentview_image: camera_rgb/images`.

        NOTE: this is distinct from get_output_data()/yaml_output_data, which
        parses an older, incompatible two-level nested convention
        (`output: <group>: {<field>: <path>}`) for UI display only and is
        never consulted by the actual conversion engine
        (yaml_driven_converter.DataTransformEngine._build_bridge_map, which
        this method mirrors exactly) or by dispatch's get_obs_config() below.
        """
        if not self.yaml_data:
            return {}
        output_section = self.yaml_data.get("config", {}).get("output", {})
        return {
            key: value
            for key, value in output_section.items()
            if key not in ("streams", "prefix_path") and isinstance(value, str)
        }

    def get_actions_to_extract(self) -> list:
        return self.yaml_actions_to_extract

    def get_input_by_name(self, name):
        for entry in self.yaml_input_data:
            if entry["name"] == name:
                return entry
        return None

    def get_train_model_name(self):
        return self.yaml_data['config']['training_model']

    # This is for the action dispatcher
    def get_obs_config(self):
        # Bridge-map entries (camera/organized-pointcloud obs, e.g.
        # `agentview_image: camera_rgb/images`) become passthrough obs in
        # obs_config.yaml — ObsBuilder only needs the source input name, not
        # the recorded dataset name (it decodes live from the topic, not
        # from an HDF5 path), so keep only the part before the first '/'.
        bridge_obs = {
            key: {"source": path.split("/")[0]}
            for key, path in self.get_output_bridge_map().items()
        }
        return {
            "config": {
                "sync":{
                    "threshold": self.yaml_data['config']['sync_tolerance_sec'],
                },
                "input": self.get_input_data_raw(),
                "observations": {**self.get_streams(), **bridge_obs},
                },
            }
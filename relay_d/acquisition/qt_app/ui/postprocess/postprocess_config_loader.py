import os
import yaml
from typing import Dict, List, Optional, Any
from relay_d.utils.coloring_logger import logger


class PostprocessConfigLoader:
    """Loads and manages postprocessing format configurations from YAML files"""

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_dir = os.path.join(script_dir, "postprocess_configs")

        self.config_dir = config_dir
        self.configs: Dict[str, Dict] = {}
        self.load_all_configs()

    def load_all_configs(self):
        """Load all YAML config files from the config directory"""
        if not os.path.exists(self.config_dir):
            logger.warning(f"Config directory does not exist: {self.config_dir}")
            return

        logger.info(f"Loading postprocessing configs from: {self.config_dir}")

        for filename in os.listdir(self.config_dir):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                self.load_config(filename)

    def load_config(self, filename: str) -> Optional[Dict]:
        """Load a single config file"""
        filepath = os.path.join(self.config_dir, filename)

        try:
            with open(filepath, "r") as f:
                config = yaml.safe_load(f)

            format_name = config.get("format", {}).get(
                "name", filename.replace(".yaml", "")
            )
            self.configs[format_name] = config

            logger.info(f"Loaded config: {format_name} from {filename}")
            return config

        except Exception as e:
            logger.error(f"Error loading config {filename}: {e}")
            return None

    def generate_html_structure_preview(self, format_name: str) -> str:
        """Generate an HTML preview of the output structure"""
        structure = self.get_output_structure(format_name)
        if not structure:
            return "<p>No structure information available</p>"

        format_info = self.get_format_info(format_name)
        html_parts = []

        html_parts.append("<html><head><style>")
        html_parts.append("""
            body {
                font-family: "JetBrains Mono", "Fira Code", monospace;
                padding: 10px;
                background-color: transparent;
                font-size: 12px;
                line-height: 1.6;
            }
            h1 { color: #1A7DC4; font-size: 14px; margin-bottom: 4px; }
            p  { color: #6B7280; margin: 2px 0; }
            strong { color: #374151; }
            .group {
                border-left: 3px solid #1A7DC4;
                padding: 6px 10px;
                margin: 6px 0;
                border-radius: 0 4px 4px 0;
            }
            .dataset {
                border-left: 3px solid #4A7C3F;
                padding: 4px 10px;
                margin: 4px 0 4px 20px;
                border-radius: 0 4px 4px 0;
            }
            .attribute {
                border-left: 3px solid #A0522D;
                padding: 4px 10px;
                margin: 4px 0 4px 40px;
                border-radius: 0 4px 4px 0;
            }
            .nested-group { margin-left: 20px; }
            .label       { font-weight: 600; color: #1A7DC4; }
            .label-ds    { font-weight: 600; color: #4A7C3F; }
            .label-attr  { font-weight: 600; color: #A0522D; }
            .description { font-style: italic; color: #6B7280; }
        """)
        html_parts.append("</style></head><body>")

        if format_info is not None:
            html_parts.append(f"<h1>{format_info['name']} Format</h1>")
            html_parts.append(f"<p><strong>Version:</strong> {format_info['version']}</p>")
            html_parts.append(f"<p><strong>Description:</strong> {format_info['description']}</p>")
        else:
            html_parts.append("<h1>Format Preview</h1>")

        html_parts.append(self._render_structure_to_html(structure))

        html_parts.append("</body></html>")

        return "\n".join(html_parts)

    def _render_structure_to_html(self, structure: Dict, indent: int = 0) -> str:
        """Recursively render structure to HTML"""
        html_parts = []
        root_groups = structure.get("root_groups", [])

        for group in root_groups:
            html_parts.append(self._render_group(group, indent))

        return "\n".join(html_parts)

    def _render_group(self, group: Dict, indent: int) -> str:
        """Render a single group to HTML"""
        indent_str = "  " * indent
        name = group.get("name", "unnamed")
        description = group.get("description", "")

        html_parts = []
        html_parts.append(f"{indent_str}<div class='group'>")
        html_parts.append(f"{indent_str}  <h3><span class='label'>{name}</span></h3>")
        if description:
            html_parts.append(f"{indent_str}  <p class='description'>{description}</p>")

        if "groups" in group:
            html_parts.append(f"{indent_str}  <div class='nested-group'>")
            for subgroup in group["groups"]:
                html_parts.append(self._render_group(subgroup, indent + 1))
            html_parts.append(f"{indent_str}  </div>")

        if "datasets" in group:
            for dataset in group["datasets"]:
                html_parts.append(self._render_dataset(dataset, indent + 1))

        if "attributes" in group:
            for attr in group["attributes"]:
                html_parts.append(self._render_attribute(attr, indent + 2))

        html_parts.append(f"{indent_str}</div>")

        return "\n".join(html_parts)

    def _render_dataset(self, dataset: Dict, indent: int) -> str:
        """Render a dataset to HTML"""
        indent_str = "  " * indent
        name = dataset.get("name", "unnamed")
        dtype = dataset.get("dtype", "unknown")
        shape = dataset.get("shape", "unknown")
        description = dataset.get("description", "")

        html_parts = []
        html_parts.append(f"{indent_str}<div class='dataset'>")
        html_parts.append(f"{indent_str}  <strong>{name}</strong> [{dtype}]")
        if description:
            html_parts.append(f"{indent_str}  <p class='description'>{description}</p>")
        html_parts.append(f"{indent_str}  <p><strong>Shape:</strong> {shape}</p>")
        html_parts.append(f"{indent_str}</div>")

        return "\n".join(html_parts)

    def _render_attribute(self, attr: Dict, indent: int) -> str:
        """Render an attribute to HTML"""
        indent_str = "  " * indent
        name = attr.get("name", "unnamed")
        dtype = attr.get("type", "unknown")
        description = attr.get("description", "")

        html_parts = []
        html_parts.append(f"{indent_str}<div class='attribute'>")
        html_parts.append(f"{indent_str}  <strong>@{name}</strong> [{dtype}]")
        if description:
            html_parts.append(f"{indent_str}  <p class='description'>{description}</p>")
        html_parts.append(f"{indent_str}</div>")

        return "\n".join(html_parts)

    ### Getters for config data ###

    def get_config(self, format_name: str) -> Optional[Dict]:
        """Get a specific config by format name"""
        return self.configs.get(format_name)

    def get_all_format_names(self) -> List[str]:
        """Get list of all available format names"""
        return list(self.configs.keys())

    def get_format_info(self, format_name: str) -> Optional[Dict[str, str]]:
        """Get basic info about a format"""
        config = self.get_config(format_name)
        if config and "format" in config:
            return {
                "name": config["format"].get("name", format_name),
                "version": config["format"].get("version", "unknown"),
                "description": config["format"].get(
                    "description", "No description available"
                ),
            }
        return None

    def get_output_structure(self, format_name: str) -> Optional[Dict]:
        """Get the output structure for a format"""
        config = self.get_config(format_name)
        return config.get("output_structure") if config else None

    def get_conversion_rules(self, format_name: str) -> Optional[Dict]:
        """Get the conversion rules for a format"""
        config = self.get_config(format_name)
        return config.get("conversion_rules") if config else None

    def get_environment_metadata_schema(self, format_name: str) -> Optional[Dict]:
        """Get the environment metadata schema for a format"""
        config = self.get_config(format_name)
        return config.get("environment_metadata") if config else None

    def get_validation_rules(self, format_name: str) -> Optional[Dict]:
        """Get the validation rules for a format"""
        config = self.get_config(format_name)
        return config.get("validation") if config else None

    def get_required_env_metadata_fields(self, format_name: str) -> List[Dict]:
        """Get required environment metadata fields"""
        env_schema = self.get_environment_metadata_schema(format_name)
        if not env_schema:
            return []

        required_fields = []
        for field in env_schema.get("required_fields", []):
            field["required"] = True
            required_fields.append(field)

        return required_fields

    def get_optional_env_metadata_fields(self, format_name: str) -> List[Dict]:
        """Get optional environment metadata fields"""
        env_schema = self.get_environment_metadata_schema(format_name)
        if not env_schema:
            return []

        optional_fields = []
        for field in env_schema.get("required_fields", []):
            if field.get("optional_fields"):
                for opt_field in field["optional_fields"]:
                    opt_field["required"] = False
                    optional_fields.append(opt_field)

        return optional_fields

    def get_output_train_config(self, format_name: str, model: str="bc_rnn_gmm") -> Optional[Dict]:
        """Get the output train config for a format"""
        return os.path.join(self.config_dir, f"{format_name}_{model}_config.json") if self.config_dir else None

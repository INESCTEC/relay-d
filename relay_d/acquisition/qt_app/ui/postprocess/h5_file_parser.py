import html
import h5py

H5_COLORS = {
    "key":   "#1A7DC4",
    "value": "#6B7280",
}

class h5FileParser:
    def __init__(self):
        pass

    def _h5_to_dict(self, group):
        """Recursively convert an h5py Group/Dataset to a nested dict."""
        result = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                result[key] = self._h5_to_dict(item)
            elif isinstance(item, h5py.Dataset):
                result[key] = f"shape={item.shape}  dtype={item.dtype}"
            else:
                result[key] = str(item)
        return result

    def _dict_to_html(self, d, indent=0):
        """Convert nested dict to colored HTML."""
        html_str = ""
        for k, v in d.items():
            pad = "&nbsp;" * (indent * 4)
            if isinstance(v, dict):
                html_str += (
                    f'{pad}<span style="color:{H5_COLORS["key"]};font-weight:600">'
                    f'{html.escape(str(k))}</span>:<br>'
                )
                html_str += self._dict_to_html(v, indent + 1)
            else:
                html_str += (
                    f'{pad}<span style="color:{H5_COLORS["key"]};font-weight:600">'
                    f'{html.escape(str(k))}</span>: '
                    f'<span style="color:{H5_COLORS["value"]}">'
                    f'{html.escape(str(v))}</span><br>'
                )
        return html_str

    def convert_h5_to_html(self, input_file):
        """Convert H5 structure to colored HTML for display."""
        with h5py.File(input_file, "r") as f:
            h5_dict = self._h5_to_dict(f)
        html_str = self._dict_to_html(h5_dict)
        return (
            '<div style="background:transparent; padding:10px; '
            'border-radius:8px; font-family:monospace; font-size:12px; '
            'line-height:1.6;">'
            f"{html_str}</div>"
        )

import argparse
import os
import shutil

import h5py
from builtin_interfaces.msg import Time

from relay_d.acquisition.qt_app.ui.config.yaml_parser import YamlParser
from relay_d.acquisition.qt_app.ui.review.demo_playback_reconstructor import DemoPlaybackReader
from relay_d.acquisition.qt_app.utils.stream_builder import StreamBuilder
from relay_d.utils.coloring_logger import logger

# Header stamps aren't read by any stream spec — a zeroed dummy is enough.
_DUMMY_STAMP = Time()

class ReprocessRecordedData:
    def __init__(self, input_dir, output_dir=None, new_config=None):
        self.input_dir = input_dir
        self.output_dir = output_dir or os.path.join(input_dir, "reprocessed")
        self.yaml_parser = self._load_new_config(new_config) if new_config else None

    def _load_new_config(self, new_config):
        parser = YamlParser()
        if not parser.load_yaml_file(new_config):
            raise ValueError(f"Failed to load config: {new_config}")
        return parser

    def reprocess(self):
        if not os.path.exists(self.input_dir):
            print(f"Input directory {self.input_dir} does not exist.")
            return

        copied_h5_paths = []

        for root, dirs, files in os.walk(self.input_dir):
            if os.path.commonpath([root, self.output_dir]) == self.output_dir:
                continue  # never recurse into an output dir from a prior run

            rel_root = os.path.relpath(root, self.input_dir)
            out_root = os.path.join(self.output_dir, rel_root) if rel_root != "." else self.output_dir
            os.makedirs(out_root, exist_ok=True)

            for file in files:
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(out_root, file)

                shutil.copy2(input_file_path, output_file_path)
                print(f"Reprocessed {input_file_path} to {output_file_path}")

                if file.endswith(".h5"):
                    copied_h5_paths.append(output_file_path)

        if self.yaml_parser:
            for h5_path in copied_h5_paths:
                self._rebuild_streams(h5_path)

    def _rebuild_streams(self, h5_path):
        builder = StreamBuilder(self.yaml_parser)
        if not builder.has_streams():
            logger.warning(f"New config defines no streams — skipping {h5_path}")
            return

        reader = DemoPlaybackReader(h5_path)
        try:
            channels_by_name = {channel.name: channel for channel in reader.channels}
            needed_inputs = builder.referenced_input_names()
            for frame_idx in range(reader.num_frames):
                all_data = {}
                for input_name in needed_inputs:
                    channel = channels_by_name.get(input_name)
                    if channel is None:
                        continue
                    msg = channel.build(frame_idx, _DUMMY_STAMP)
                    if msg is not None:
                        all_data[input_name] = msg
                builder.evaluate(all_data)
        finally:
            reader.close()

        new_streams = builder.finalize()

        # Rebuild into a fresh file rather than deleting "streams" in place: HDF5
        # stores these fixed-shape chunked+compressed datasets with a "Fixed Array"
        # chunk index, and unlinking such a dataset hits a longstanding HDF5 bug
        # ("bad version number for layout message") that corrupts the file handle.
        tmp_path = h5_path + ".tmp"
        try:
            with h5py.File(h5_path, "r") as src, h5py.File(tmp_path, "w") as dst:
                dst.attrs.update(src.attrs)
                demo_name = next(iter(src["data"].keys()))

                dst_data_group = dst.create_group("data")
                for name, item in src["data"].items():
                    if name != demo_name:
                        dst_data_group.copy(item, name)
                        continue

                    dst_demo_group = dst_data_group.create_group(name)
                    dst_demo_group.attrs.update(item.attrs)
                    for child_name, child in item.items():
                        if child_name == "streams":
                            continue
                        dst_demo_group.copy(child, child_name)

                for name, item in src.items():
                    if name != "data":
                        dst.copy(item, name)

                streams_grp = dst_demo_group.create_group("streams")
                for stream_name, arr in new_streams.items():
                    streams_grp.create_dataset(
                        stream_name, data=arr, compression="gzip", compression_opts=4
                    )
                    logger.info(f"Rebuilt stream '{stream_name}' in {h5_path}: shape {arr.shape}")
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        os.replace(tmp_path, h5_path)


def main():
    parser = argparse.ArgumentParser(
        description="Copy a recorded dataset and, optionally, rebuild its streams/ "
        "group from a new output.streams config, reading from the raw inputs "
        "already present in each recorded .h5 file."
    )
    parser.add_argument("--input_dir", help="Path to the raw recorded dataset directory")
    parser.add_argument("--output_dir", help="Path to the output directory for reprocessed data")
    parser.add_argument(
        "--config",
        dest="new_config",
        default=None,
        help="Path to a new config YAML whose config.output.streams (and config.input "
        "joint_names filters) should be used to rebuild streams/ in the copied dataset",
    )
    args = parser.parse_args()

    reprocessor = ReprocessRecordedData(args.input_dir, args.output_dir, args.new_config)
    reprocessor.reprocess()


if __name__ == "__main__":
    main()

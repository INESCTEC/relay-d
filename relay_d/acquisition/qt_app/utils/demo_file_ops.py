import os

from relay_d.utils.coloring_logger import logger


def delete_demo_file(file_path: str, recorder=None) -> bool:
    """Delete a recorded demo .h5 file from disk. Returns True on success.

    Refused for the file `recorder` is still writing (recording or saving): Linux would remove it while open,
    and Stop would then report the lost demo as saved."""
    if recorder is not None and getattr(recorder, "h5_file", None) is not None and \
            os.path.abspath(file_path or "") == os.path.abspath(getattr(recorder, "h5_file_path", None) or ""):
        logger.error(f"Not deleting {file_path}: it is the demo being recorded. Stop the recording first.")
        return False
    try:
        if file_path and os.path.isfile(file_path):
            os.remove(file_path)
            logger.info(f"Deleted demo file: {file_path}")
            return True
        logger.warning(f"Demo file not found, nothing to delete: {file_path}")
        return False
    except Exception as e:
        logger.error(f"Error deleting demo file {file_path}: {e}")
        return False

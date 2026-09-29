import os

from relay_d.utils.coloring_logger import logger


def delete_demo_file(file_path: str) -> bool:
    """Delete a recorded demo .h5 file from disk. Returns True on success."""
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

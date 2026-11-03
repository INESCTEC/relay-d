#!/usr/bin/env python3
"""Headless dataset conversion (no GUI, no ROS). See relay_d/acquisition/convert_cli.py.

    python scripts/convert_dataset.py --config acquisition.yaml --format lerobot \
        --input recordings/ --output my_lerobot_dataset
"""

import sys

from relay_d.acquisition.convert_cli import main

if __name__ == "__main__":
    sys.exit(main())

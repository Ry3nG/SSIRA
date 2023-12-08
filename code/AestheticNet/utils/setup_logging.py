import logging
import os

def setup_logging(target_path: str, current_time: str) -> None:
    log_file = os.path.join(target_path, f"train_{current_time}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
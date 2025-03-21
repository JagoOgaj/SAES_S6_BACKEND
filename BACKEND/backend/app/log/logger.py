import logging
from logging.handlers import RotatingFileHandler
from backend.app.core.const import ENUM_LOGGER_ENV
import os
from dotenv import load_dotenv

load_dotenv()


def setup_logger():
    logger = logging.getLogger("app_logger")
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)

    log_file_path = os.environ.get(ENUM_LOGGER_ENV.LOG_FILE_PATH.value)
    log_file_max_bytes = int(os.environ.get(ENUM_LOGGER_ENV.LOG_FILE_MAX_BYTES.value))
    log_file_backup_count = int(
        os.environ.get(ENUM_LOGGER_ENV.LOG_FILE_BACKUP_COUNT.value)
    )

    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=log_file_max_bytes, backupCount=log_file_backup_count
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()

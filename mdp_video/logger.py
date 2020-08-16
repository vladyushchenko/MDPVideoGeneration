"""
Copyright (C) 2017 NVIDIA Corporation.

All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""
import logging
import os
from datetime import datetime
from typing import Any

from tensorboardX import SummaryWriter


def set_logger() -> logging.Logger:
    """
    Set root logger.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger


class Logger:
    """
    Logger class for application.
    """

    logger = set_logger()
    time = datetime.now()

    def __init__(self, log_dir: str) -> None:
        """
        Init call.

        :param log_dir: logging dir
        """
        logger_dir = os.path.join(log_dir, "Logs")
        if not os.path.exists(logger_dir):
            os.makedirs(logger_dir)

        filename = "{}_{}.log".format(__name__, self.time.isoformat())
        file_path = os.path.join(logger_dir, filename)
        if os.path.exists(file_path):
            return

        log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(module)s - %(message)s")
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(log_formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.logger.addHandler(console_handler)

        self.summary = SummaryWriter(log_dir)

    def scalar_summary(self, tag: str, value: Any, step: int) -> None:
        """
        Save losses summary to tensorboard.
        """
        self.summary.add_scalar(tag, value, step)

    def histogram_summary(self, tag: str, tensor: Any, step: int) -> None:
        """
        Save weight histogram summary to tensorboard.
        """
        self.summary.add_histogram(tag, tensor, step)

    def image_summary(self, tag: str, images: Any, iteration: int) -> None:
        """
        Save image summary to tensorboard.
        """
        for img in images:
            self.summary.add_image(tag, img.transpose(2, 0, 1), iteration)
            self.summary.file_writer.flush()

    def video_summary(self, tag: str, videos: Any, iteration: int) -> None:
        """
        Save video summary to tensorboard.
        """
        self.summary.add_video(tag, videos, iteration)
        self.summary.file_writer.flush()

    def log(self, message: str, level: int = logging.INFO) -> None:
        """
        Translate message to root logger.
        """
        self.logger.log(level=level, msg=message)

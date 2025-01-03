# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import json
import logging
import logging.config
import os
import re
from logging import Logger
from logging.handlers import RotatingFileHandler

from nvflare.apis.workspace import Workspace


class ANSIColor:
    # Basic ANSI color codes
    COLORS = {
        "black": "30",
        "red": "31",
        "bold_red": "31;1",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        "white": "37",
        "grey": "38",
        "reset": "0",
    }

    # Default logger level:color mappings
    DEFAULT_LEVEL_COLORS = {
        "NOTSET": COLORS["grey"],
        "DEBUG": COLORS["grey"],
        "INFO": COLORS["grey"],
        "WARNING": COLORS["yellow"],
        "ERROR": COLORS["red"],
        "CRITICAL": COLORS["bold_red"],
    }

    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """Wrap text with the given ANSI SGR color.

        Args:
            text (str): text to colorize.
            color (str): ANSI SGR color code or color name defined in ANSIColor.COLORS.

        Returns:
            colorized text
        """
        if not any(c.isdigit() for c in color):
            color = cls.COLORS.get(color.lower(), cls.COLORS["reset"])

        return f"\x1b[{color}m{text}\x1b[{cls.COLORS['reset']}m"


class BaseFormatter(logging.Formatter):
    def __init__(self, fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt=None, style="%"):
        """Default formatter for log records.

        Shortens logger %(name)s to the basenames. Full name can be accessed with %(fullName)s

        Args:
            fmt (str): format string which uses LogRecord attributes.
            datefmt (str): date/time format string. Defaults to '%Y-%m-%d %H:%M:%S'.
            style (str): style character '%' '{' or '$' for format string.

        """
        self.fmt = fmt
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)

    def format(self, record):
        if not hasattr(record, "fullName"):
            record.fullName = record.name
            record.name = record.name.split(".")[-1]

        return super().format(record)


class ColorFormatter(BaseFormatter):
    def __init__(
        self,
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt=None,
        style="%",
        level_colors=ANSIColor.DEFAULT_LEVEL_COLORS,
        logger_colors={},
    ):
        """Format colors based on log levels. Optionally can provide mapping based on logger namess.

        Args:
            fmt (str): format string which uses LogRecord attributes.
            datefmt (str): date/time format string. Defaults to '%Y-%m-%d %H:%M:%S'.
            style (str): style character '%' '{' or '$' for format string.
            level_colors (Dict[str, str]): dict of levelname: ANSI color. Defaults to ANSIColor.DEFAULT_LEVEL_COLORS.
            logger_colors (Dict[str, str]): dict of loggername: ANSI color. Defaults to {}.

        """
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.level_colors = level_colors
        self.logger_colors = logger_colors

    def format(self, record):
        super().format(record)

        # Apply level_colors based on record levelname
        log_color = self.level_colors.get(record.levelname, "reset")

        # Apply logger_color to logger_names if INFO or below
        if record.levelno <= logging.INFO:
            log_color = self.logger_colors.get(record.name, log_color)

        log_fmt = ANSIColor.colorize(self.fmt, log_color)

        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class JsonFormatter(BaseFormatter):
    def __init__(
        self,
        fmt="%(asctime)s - %(name)s - %(fullName)s - %(levelname)s - %(message)s",
        datefmt=None,
        style="%",
        extract_brackets=True,
    ):
        """Format log records into JSON.

        Args:
            fmt (str): format string which uses LogRecord attributes. Attributes are used for JSON keys.
            datefmt (str): date/time format string. Defaults to '%Y-%m-%d %H:%M:%S'.
            style (str): style character '%' '{' or '$' for format string.
            extract_bracket_fields (bool): whether to extract bracket fields of message into sub-dictionary. Defaults to True.

        """
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.fmt_dict = self.generate_fmt_dict(self.fmt)
        self.extract_brackets = extract_brackets

    def generate_fmt_dict(self, fmt: str) -> dict:
        # Parse the `fmt` string and create a mapping of keys to LogRecord attributes
        matches = re.findall(r"%\((.*?)\)([sd])", fmt)

        fmt_dict = {}
        for key, _ in matches:
            if key == "shortname":
                fmt_dict["name"] = "shortname"
            else:
                fmt_dict[key] = key

        return fmt_dict

    def extract_bracket_fields(self, message: str) -> dict:
        # Extract bracketed fl_ctx_fields eg. [k1=v1, k2=v2...] into sub-dictionary
        bracket_fields = {}
        match = re.search(r"\[(.*?)\]:", message)
        if match:
            pairs = match.group(1).split(", ")
            for pair in pairs:
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    bracket_fields[key] = value
        return bracket_fields

    def formatMessage(self, record) -> dict:
        return {fmt_key: record.__dict__.get(fmt_val, "") for fmt_key, fmt_val in self.fmt_dict.items()}

    def format(self, record) -> str:
        super().format(record)

        record.message = record.getMessage()
        bracket_fields = self.extract_bracket_fields(record.message) if self.extract_brackets else None
        record.asctime = self.formatTime(record)

        formatted_message_dict = self.formatMessage(record)
        message_dict = {k: v for k, v in formatted_message_dict.items() if k != "message"}

        if bracket_fields:
            message_dict["fl_ctx_fields"] = bracket_fields
            record.message = re.sub(r"\[.*?\]:", "", record.message).strip()

        message_dict[self.fmt_dict.get("message", "message")] = record.message

        return json.dumps(message_dict, default=str)


class LoggerNameFilter(logging.Filter):
    def __init__(self, logger_names=["nvflare"]):
        """Filter log records based on logger names.

        Args:
            logger_names (List[str]): list of logger names to allow through filter (inclusive)

        """
        super().__init__()
        self.logger_names = logger_names

    def filter(self, record):
        name = record.fullName if hasattr(record, "fullName") else record.name
        return any(name.startswith(logger_name) for logger_name in self.logger_names)


def get_module_logger(module=None, name=None):
    if module is None:
        caller_globals = inspect.stack()[1].frame.f_globals
        module = caller_globals.get("__name__", "")

    return logging.getLogger(f"{module}.{name}" if name else module)


def get_obj_logger(obj):
    return logging.getLogger(f"{obj.__module__}.{obj.__class__.__qualname__}")


def get_script_logger():
    # Get script logger name based on filename and package. If not in a package, default to custom.
    caller_frame = inspect.stack()[1]
    package = caller_frame.frame.f_globals.get("__package__", "")
    file = caller_frame.frame.f_globals.get("__file__", "")

    return logging.getLogger(
        f"{package if package else 'custom'}{'.' + os.path.splitext(os.path.basename(file))[0] if file else ''}"
    )


def configure_logging(workspace: Workspace, dir_path: str = "", file_prefix: str = ""):
    # Read log_config.json from workspace, update with file_prefix, and apply to dir_path
    log_config_file_path = workspace.get_log_config_file_path()
    assert os.path.isfile(log_config_file_path), f"missing log config file {log_config_file_path}"

    with open(log_config_file_path, "r") as f:
        dict_config = json.load(f)

    apply_log_config(dict_config, dir_path, file_prefix)


def apply_log_config(dict_config, dir_path: str = "", file_prefix: str = ""):
    # Update log config dictionary with file_prefix, and apply to dir_path
    stack = [dict_config]
    while stack:
        current_dict = stack.pop()
        for key, value in current_dict.items():
            if isinstance(value, dict):
                stack.append(value)
            elif key == "filename":
                if file_prefix:
                    value = os.path.join(os.path.dirname(value), file_prefix + "_" + os.path.basename(value))
                current_dict[key] = os.path.join(dir_path, value)

    logging.config.dictConfig(dict_config)


def add_log_file_handler(log_file_name):
    root_logger = logging.getLogger()
    main_handler = root_logger.handlers[0]
    file_handler = RotatingFileHandler(log_file_name, maxBytes=20 * 1024 * 1024, backupCount=10)
    file_handler.setLevel(main_handler.level)
    file_handler.setFormatter(main_handler.formatter)
    root_logger.addHandler(file_handler)


def print_logger_hierarchy(package_name="nvflare", level_colors=ANSIColor.DEFAULT_LEVEL_COLORS):
    all_loggers = logging.root.manager.loggerDict

    # Filter for package loggers based on package_name
    package_loggers = {name: logger for name, logger in all_loggers.items() if name.startswith(package_name)}
    sorted_package_loggers = sorted(package_loggers.keys())

    # Print package loggers with hierarcjy
    print(f"hierarchical loggers ({len(package_loggers)}):")

    def get_effective_level(logger_name):
        # Search for effective level from parent loggers
        parts = logger_name.split(".")
        for i in range(len(parts), 0, -1):
            parent_name = ".".join(parts[:i])
            parent_logger = package_loggers.get(parent_name)
            if isinstance(parent_logger, Logger) and parent_logger.level != logging.NOTSET:
                return logging.getLevelName(parent_logger.level)

        # If no parent has a set level, default to the root logger's effective level
        return logging.getLevelName(logging.root.level)

    def print_hierarchy(logger_name, indent_level=0):
        logger = package_loggers.get(logger_name)
        level_name = get_effective_level(logger_name)

        # Indicate "(unset)" placeholders if logger.level == NOTSET
        is_unset = isinstance(logger, Logger) and logger.level == logging.NOTSET or not isinstance(logger, Logger)
        level_display = f"{level_name} (SET)" if not is_unset else level_name

        # Print the logger with color and indentation
        color = level_colors.get(level_name, ANSIColor.COLORS["reset"])
        print("    " * indent_level + ANSIColor.colorize(f"{logger_name} [{level_display}]", color))

        # Find child loggers based on the current hierarchy level
        for name in sorted_package_loggers:
            if name.startswith(logger_name + ".") and name.count(".") == logger_name.count(".") + 1:
                print_hierarchy(name, indent_level + 1)

    print_hierarchy(package_name)

# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import os
import sys
from datetime import datetime
from enum import Enum
from typing import Any, NamedTuple, Dict, List, Tuple, Optional, Union

from nvflare.common.settings import SettingsReader

FLARE_LOG_LEVEL = "flare.log_level"
FLARE_LOG_LEVELS = "flare.log_levels"
FLARE_PROFILES = "flare_profiles"
EXTENSIONS = ["yml", "yaml", "json"]

log = logging.getLogger(__name__)
SourceType = Enum("SourceType", "FLAT DICT")


class SettingsSource(NamedTuple):
    source_type: SourceType
    source_name: str
    modified_time: datetime
    data: Dict[str, Any]


def canonicalize(name: str) -> str:
    return ''.join(c.lower() for c in name if c.isalnum())


def append_arg(value: Any, new_value: str) -> Union[str, List[str]]:
    if not value:
        return new_value

    if type(value) is list:
        value.append(new_value)
        return value
    else:
        return [value, new_value]


def load_environment_vars() -> SettingsSource:
    data = dict((canonicalize(key), value) for key, value in os.environ.items())
    return SettingsSource(SourceType.FLAT, "env-var", datetime.now(), data)


def load_arguments() -> SettingsSource:
    # Convert argv list into a dict
    data = {}
    last_key = None
    last_value = None

    for arg in sys.argv:
        if arg.startswith(("-", "--")):
            if last_key:
                data[last_key] = append_arg(data.get(last_key), last_value)
            if "=" in arg:
                # Handling --key=value format
                index = arg.index("=")
                last_key = canonicalize(arg[:index])
                last_value = arg[index+1:]
            else:
                last_key = canonicalize(arg)
                last_value = None
        else:
            if not last_value:
                last_value = arg

    if last_key:
        data[last_key] = append_arg(data.get(last_key), last_value)

    return SettingsSource(SourceType.FLAT, "cmd-line", datetime.now(), data)


"""Base class for settings 
"""


class Settings:

    # Those settings don't change so make them class static
    env_vars = load_environment_vars()
    cmd_args = load_arguments()

    def __init__(self, sources: List[SettingsSource] = None):
        """Construct a Settings with a list of sources

        Args:
            sources: List of sources, optional
        """
        self._sources = []
        for s in sources or []:
            self.add_source(s)

    def __getitem__(self, key):
        return self.get(key)

    def get(self, key: str, default_value: Any = None) -> Any:
        """Get the settings parameter given a key

        Args:
            key: The key in dot-notation
            default_value: Default value to return if key is not found

        Returns:
            The parameter value or default value
        """
        source_name, modified_time, value = self.get_detail(key, default_value)
        if value:
            log.debug(f"Parameter [{key}={value}] found in {source_name} last modified on {modified_time.isoformat()}")

        return value

    def get_number(self, key: str, default_value: Union[int, float, None] = None) -> Union[int, float, None]:
        """Get the settings parameter as a number given a key

        Args:
            key: The key in dot-notation
            default_value: Default value to return if key is not found

        Returns:
            The parameter value or default value.
        """
        value = self.get(key, None)
        if isinstance(value, (int, float)):
            return value

        if value:
            try:
                result = int(value)
            except ValueError:
                try:
                    result = float(value)
                except ValueError:
                    log.debug(f"Can't convert {key}={value} to number")
                    result = None

        return result if result else default_value

    def get_detail(self, key: str, default_value: Any = None) -> Tuple[str, datetime, Any]:
        """Get the settings parameter given a key with detailed information

        Args:
            key: The key in dot-notation
            default_value: Default value to return if key is not found

        Returns:
            A tuple with source_name, modified_time and the value
        """
        for source in self._sources:
            if source.source_type == SourceType.FLAT:
                value = source.data.get(canonicalize(key))
            else:
                value = Settings._get_by_dot_notation(source.data, key)

            if value:
                return source.source_name, source.modified_time, value

        if default_value:
            return "default-value", datetime.now(), default_value
        else:
            return "not-found", datetime.now(), None

    def exists(self, key: str) -> bool:
        """Check if a key is defined anywhere. This is mainly used for command-line arguments without value. The get
           call can't distinguish between empty and non-existent parameters since both return None

        Args:
            key: The key in dot-notation

        Returns:
            True if key exists in any of the sources, regardless of the value, which could be None
        """
        for source in self._sources:
            if source.source_type == SourceType.FLAT:
                result = canonicalize(key) in source.data
            else:
                result = Settings._get_by_dot_notation(source.data, key) is not None

            if result:
                return True

        return False

    def add_source(self, source: SettingsSource) -> None:
        """Add source to the Settings if it's not None

        Args:
            source: The settings source
        """
        if source:
            self._sources.append(source)

    @staticmethod
    def _find(source: SettingsSource, key: str) -> Any:
        if source.source_type == SourceType.FLAT:
            return source.data.get(canonicalize(key))
        else:
            return Settings._get_by_dot_notation(source.data, key)

    @staticmethod
    def _get_by_dot_notation(data: dict, key: str) -> Any:
        parts = key.split(".")
        parent = data
        for i in range(0, len(parts)-1):
            node = parent.get(parts[i])
            if type(node) is not dict:
                return None
            parent = node
        return parent.get(parts[-1])


"""The settings that contains environment variables and command-line arguments. This is mainly used
to find the location of configuration files so FlareSettings can be initialized.
"""


class RuntimeSettings(Settings):

    def __init__(self):
        """Construct a Settings with run-time parameters (env-var and cmd-line)
        """
        super().__init__([Settings.env_vars, Settings.cmd_args])


"""The main settings class with all sources. All configurable parameter should be retrieved using this class.
"""


class FlareSettings(RuntimeSettings):

    def __init__(self, settings_file, *config_files):
        """Construct a Settings with all sources, run-time, settings and configuration files

        Args:
            settings_file: The settings file name with absolute path. If no extension is given, all supported
                           extensions will be tried.
            config_files: List of configuration files with absolute path. Save as above, extension is optional
        """
        super().__init__()

        profiles = self.get(FLARE_PROFILES)
        settings_base, settings_ext = os.path.splitext(settings_file)

        if profiles:
            profile_list = profiles.split(",")
            log.info(f"Active profiles in use: {profile_list}")
            for p in profile_list:
                filename = settings_base + "_" + p
                source = FlareSettings._load_file(filename)
                if source:
                    self.add_source(source)
                else:
                    log.error(f"No settings file found for profile {p}. Profile is ignored")

        self.add_source_file(settings_file)

        for config in config_files:
            self.add_source_file(config)

    def add_source_file(self, file: str) -> None:
        source = FlareSettings._load_file(file)
        if source:
            log.debug(f"Configuration file {source.source_name} is loaded into Flare settings")
            self.add_source(source)

    def set_log_levels(self):

        level = self.get(FLARE_LOG_LEVEL)
        if level:
            FlareSettings._set_log_level(None, level)

        levels = self.get(FLARE_LOG_LEVELS)
        if type(levels) is dict:
            for logger, level in levels.items():
                FlareSettings._set_log_level(logger, level)

    @staticmethod
    def _load_file(file: str) -> Optional[SettingsSource]:

        filename = FlareSettings._find_file(os.path.expanduser(file))
        if not filename:
            log.debug(f"Configuration file {filename} doesn't exist")
            return None

        try:
            data = SettingsReader(filename).read()
            modified_time = datetime.fromtimestamp(os.path.getmtime(filename))
            return SettingsSource(SourceType.DICT, filename, modified_time, data)
        except IOError as e:
            log.error(f"Can't read file {filename}: {str(e)}")
            return None

    @staticmethod
    def _try_extensions(base_file: str) -> Optional[str]:

        for ext in EXTENSIONS:
            filename = base_file + "." + ext
            if os.path.isfile(filename):
                return filename

        return None

    @staticmethod
    def _find_file(file: str) -> Optional[str]:
        _, ext = os.path.splitext(file)
        if ext:
            return file if os.path.isfile(file) else None
        else:
            return FlareSettings._try_extensions(file)

    @staticmethod
    def _set_log_level(logger_name: Optional[str], level: str) -> None:

        if logger_name:
            logger = logging.getLogger(logger_name)
        else:
            logger = logging.getLogger()

        numeric_level = getattr(logging, level.upper(), None)
        if numeric_level:
            if logger:
                logger.setLevel(numeric_level)
            else:
                log.error(f"Logger {logger_name} not found")
        else:
            log.error(f"Invalid log level {level} for logger {logger_name if logger_name else 'Root'}")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%d/%b/%Y %H:%M:%S",
        stream=sys.stdout)

    runtime = RuntimeSettings()
    print(f"Profile={runtime['flare_profiles']}")

    settings = FlareSettings("~/.nvflare/admin", "/Users/zhihongz/nvflare-test/poc/admin/startup/fed_admin.json")
    settings.set_log_levels()
    print(settings["flare.log_levels"])

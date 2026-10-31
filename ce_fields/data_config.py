#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment-aware data path configuration for ce_fields.

Created on Wed Jul 1 2026
@author: Julien Emile-Geay
"""

import yaml
import os
from pathlib import Path
import socket

LOCAL_CONFIG_NAME = 'data_config.local.yml'
LOCAL_CONFIG_ENV = 'CE_FIELDS_LOCAL_CONFIG'
PLACEHOLDER_PREFIX = '/path/to/'


def _deep_merge(base: dict, override: dict) -> dict:
    """Return `base` with `override` merged in, recursing into nested dicts."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class DataConfig:
    """Environment-aware lookup of data file locations.

    Reads a `data_config.yml` (committed, machine-independent; by default the
    one shipped inside this package) and merges in, key by key, the
    machine-specific overrides found in

    1. a `data_config.local.yml` next to it (gitignored), then
    2. the file named by the CE_FIELDS_LOCAL_CONFIG environment variable, if
       set -- the way to configure a pip-installed copy of ce_fields.

    Locations of external datasets belong in such an override file, e.g.::

        environments:
          laptop:
            paths:
              lmr_data: "/my/data/LMR"

    An environment `root` that is relative (e.g. ".") is resolved against the
    directory holding `data_config.yml`.
    """

    def __init__(self, config_path=None):
        # If no path provided, use the data_config.yml shipped with the package
        if config_path is None:
            config_path = Path(__file__).resolve().parent / 'data_config.yml'
        config_path = Path(config_path).resolve()
        self.config_dir = config_path.parent

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        override_paths = [self.config_dir / LOCAL_CONFIG_NAME]
        if os.getenv(LOCAL_CONFIG_ENV):
            override_paths.append(Path(os.getenv(LOCAL_CONFIG_ENV)).expanduser())
        for override_path in override_paths:
            if override_path.exists():
                with open(override_path, 'r') as f:
                    self.config = _deep_merge(self.config, yaml.safe_load(f) or {})

        # Auto-detect environment based on hostname or override
        self.environment = self._detect_environment()

    def _detect_environment(self):
        """Auto-detect environment based on hostname or environment variables"""
        hostname = socket.gethostname()

        # Check for HPC indicators
        if any(hpc_indicator in hostname.lower() for hpc_indicator in
               ['login', 'compute', 'cluster', 'hpc']):
            return 'hpc'

        # Check environment variable override
        env_override = os.getenv('DATA_ENV')
        if env_override and env_override in self.config['environments']:
            return env_override

        # Default to configured environment
        return self.config.get('current_environment', 'laptop')

    def get_file_path(self, file_key):
        """Get full path for a data file

        Parameters
        ----------
        file_key : str
            Key identifying the file in the configuration

        Returns
        -------
        str or Path
            For URLs: returns the URL string as-is
            For absolute paths: returns Path object of the absolute path
            For relative paths: returns Path object joined with environment root

        Raises
        ------
        FileNotFoundError
            If the path still contains a "/path/to/" placeholder, i.e. the
            dataset's location has not been set for this machine.
        """
        if file_key not in self.config['files']:
            raise KeyError(f"File '{file_key}' not found in configuration")

        file_info = self.config['files'][file_key]
        raw_path = file_info['path']

        # Check if it's a URL
        if raw_path.startswith(('http://', 'https://')):
            return raw_path

        env_config = self.config['environments'][self.environment]
        path = raw_path.format(**env_config['paths'])

        if PLACEHOLDER_PREFIX in path:
            raise FileNotFoundError(
                f"No location set for '{file_key}' ({file_info.get('description', '')}) "
                f"in the '{self.environment}' environment: got placeholder '{path}'. "
                f"Download the dataset and set its root in {self.config_dir / LOCAL_CONFIG_NAME}, "
                f"or in a file named by the {LOCAL_CONFIG_ENV} environment variable (see the README)."
            )

        # Absolute paths (or roots) are used as-is; relative ones are joined
        # with the environment root, itself relative to the repository root.
        if Path(path).is_absolute():
            return Path(path)
        root = Path(env_config.get('root', '.'))
        if not root.is_absolute():
            root = self.config_dir / root
        return root / path

    def get_all_paths(self):
        """Get all configured file paths for current environment"""
        return {key: self.get_file_path(key) for key in self.config['files']}

    def list_files(self):
        """List all configured files with descriptions"""
        for key, info in self.config['files'].items():
            try:
                path = self.get_file_path(key)
            except FileNotFoundError:
                print(f"? {key}: location not set (see {LOCAL_CONFIG_NAME})")
                continue
            # Check if path is a URL (string) or Path object
            if isinstance(path, str):
                exists = "\U0001F310"  # URL indicator
            else:
                exists = "✓" if path.exists() else "✗"
            print(f"{exists} {key}: {path}")
            if 'description' in info:
                print(f"    {info['description']}")


# Usage example
if __name__ == "__main__":
    config = DataConfig()

    print(f"Current environment: {config.environment}")
    print("\nConfigured files:")
    config.list_files()

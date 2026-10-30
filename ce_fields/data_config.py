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

class DataConfig:
    def __init__(self, config_path=None):
        # If no path provided, find data_config.yml in repository root
        if config_path is None:
            # Start from this file's location and go up to find the repo root
            current_file = Path(__file__).resolve()
            repo_root = current_file.parent.parent  # Go up from ce_fields/ to repo root
            config_path = repo_root / 'data_config.yml'

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

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
        """
        if file_key not in self.config['files']:
            raise KeyError(f"File '{file_key}' not found in configuration")

        file_info = self.config['files'][file_key]
        raw_path = file_info['path']

        # Check if it's a URL
        if raw_path.startswith(('http://', 'https://')):
            return raw_path

        # Check if it's an absolute path
        if raw_path.startswith('/'):
            return Path(raw_path)

        # It's a relative path - format with environment-specific paths
        env_config = self.config['environments'][self.environment]
        relative_path = raw_path.format(**env_config['paths'])
        full_path = Path(env_config['root']) / relative_path

        return full_path

    def get_all_paths(self):
        """Get all configured file paths for current environment"""
        return {key: self.get_file_path(key) for key in self.config['files']}

    def list_files(self):
        """List all configured files with descriptions"""
        for key, info in self.config['files'].items():
            path = self.get_file_path(key)
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

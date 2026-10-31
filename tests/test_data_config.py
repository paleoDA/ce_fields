"""
Tests for DataConfig: repo-relative roots, the gitignored data_config.local.yml
override, and the error raised for datasets whose location is unset.
"""

from pathlib import Path

import pytest
import yaml

from ce_fields.data_config import DataConfig, LOCAL_CONFIG_ENV, LOCAL_CONFIG_NAME

BASE_CONFIG = {
    'environments': {
        'laptop': {
            'root': '.',
            'paths': {'results': 'results', 'ext_data': '/path/to/ext'},
        },
    },
    'files': {
        'table': {'path': '{results}/table.csv', 'description': 'in-repo table'},
        'ext_file': {'path': '{ext_data}/field.nc', 'description': 'external field'},
        'remote': {'path': 'https://example.org/data.nc'},
    },
    'current_environment': 'laptop',
}


def _write_config(directory: Path, local: dict = None) -> Path:
    config_path = directory / 'data_config.yml'
    config_path.write_text(yaml.safe_dump(BASE_CONFIG))
    if local is not None:
        (directory / LOCAL_CONFIG_NAME).write_text(yaml.safe_dump(local))
    return config_path


@pytest.fixture(autouse=True)
def laptop_env(monkeypatch):
    monkeypatch.setenv('DATA_ENV', 'laptop')
    monkeypatch.setattr('socket.gethostname', lambda: 'my-laptop')


def test_relative_root_resolves_to_config_directory(tmp_path):
    config = DataConfig(_write_config(tmp_path))
    assert config.get_file_path('table') == tmp_path / 'results' / 'table.csv'


def test_urls_are_returned_unchanged(tmp_path):
    config = DataConfig(_write_config(tmp_path))
    assert config.get_file_path('remote') == 'https://example.org/data.nc'


def test_placeholder_raises_with_pointer_to_local_file(tmp_path):
    config = DataConfig(_write_config(tmp_path))
    with pytest.raises(FileNotFoundError, match=LOCAL_CONFIG_NAME):
        config.get_file_path('ext_file')


def test_local_override_sets_external_root_and_keeps_other_entries(tmp_path):
    local = {'environments': {'laptop': {'paths': {'ext_data': '/data/ext'}}}}
    config = DataConfig(_write_config(tmp_path, local))
    assert config.get_file_path('ext_file') == Path('/data/ext/field.nc')
    # Entries absent from the local file are untouched by the merge
    assert config.get_file_path('table') == tmp_path / 'results' / 'table.csv'


def test_env_var_override_is_applied_after_local_file(tmp_path, monkeypatch):
    local = {'environments': {'laptop': {'paths': {'ext_data': '/data/from_local'}}}}
    env_file = tmp_path / 'elsewhere.yml'
    env_file.write_text(yaml.safe_dump(
        {'environments': {'laptop': {'paths': {'ext_data': '/data/from_env'}}}}))
    monkeypatch.setenv(LOCAL_CONFIG_ENV, str(env_file))
    config = DataConfig(_write_config(tmp_path, local))
    assert config.get_file_path('ext_file') == Path('/data/from_env/field.nc')


def test_packaged_config_is_found_by_default():
    config = DataConfig()
    assert (config.config_dir / 'data_config.yml').exists()
    assert any(key.startswith('pmip4_') for key in config.config['files'])


def test_list_files_reports_unset_locations(tmp_path, capsys):
    DataConfig(_write_config(tmp_path)).list_files()
    assert 'ext_file: location not set' in capsys.readouterr().out

import os
import tempfile
import yaml
from unittest.mock import patch, mock_open
from dvc import config
from scilpy.io.dvc import (
    get_default_config,
    pull_test_case_package,
    pull_package_from_dvc_repository,
)


def test_get_default_config():
    assert isinstance(get_default_config(), config.Config)


@patch('scilpy.io.dvc.api.DVCFileSystem')
@patch('dvc.config.Config.load_one', return_value={'remote': {'scil-data': {'url': 'test_url'}}})
def test_pull_test_case_package(mock_load_one, mock_dvc_fs):
    with tempfile.TemporaryDirectory() as d:
        with patch('scilpy.SCILPY_HOME', d):
            test_descriptors = {"test_package": {"revision": "123"}}
            with patch('builtins.open', new_callable=mock_open,
                       read_data=yaml.dump(test_descriptors)):
                pull_test_case_package("test_package")
                mock_dvc_fs.assert_called_once()


@patch('scilpy.io.dvc.api.DVCFileSystem')
@patch('dvc.config.Config.load_one', return_value={'remote': {'scil-data': {'url': 'test_url'}}})
def test_pull_package_from_dvc_repository(mock_load_one, mock_dvc_fs):
    with tempfile.TemporaryDirectory() as d:
        pull_package_from_dvc_repository(
            "vendor", "package", d)
        mock_dvc_fs.assert_called_once()

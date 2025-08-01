import os
import tempfile
import zipfile
from unittest.mock import patch, MagicMock
from scilpy.io.fetcher import (
    download_file_from_google_drive,
    get_testing_files_dict,
    fetch_data,
    get_synb0_template_path,
)


@patch('requests.Session.get')
def test_download_file_from_google_drive(mock_get):
    with tempfile.NamedTemporaryFile() as f:
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b'test']
        mock_get.return_value = mock_response
        download_file_from_google_drive("http://fake.url", f.name)
        with open(f.name, 'rb') as downloaded_file:
            assert downloaded_file.read() == b'test'


def test_get_testing_files_dict():
    files_dict = get_testing_files_dict()
    assert isinstance(files_dict, dict)
    assert "commit_amico.zip" in files_dict


@patch('scilpy.io.fetcher.download_file_from_google_drive')
@patch('hashlib.md5')
@patch('zipfile.ZipFile')
def test_fetch_data(mock_zipfile, mock_md5, mock_download):
    with tempfile.TemporaryDirectory() as d:
        with patch('scilpy.SCILPY_HOME', d):
            files_dict = {"test.zip": "d41d8cd98f00b204e9800998ecf8427e"}
            mock_md5.return_value.hexdigest.return_value = "d41d8cd98f00b204e9800998ecf8427e"

            def create_dummy_file(url, dest):
                with open(dest, 'w') as f:
                    f.write("test")
            mock_download.side_effect = create_dummy_file
            fetch_data(files_dict)
            mock_download.assert_called_once()
            mock_zipfile.assert_called_once()


def test_get_synb0_template_path():
    path = get_synb0_template_path()
    assert os.path.exists(path)

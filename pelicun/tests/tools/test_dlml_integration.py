#
# Copyright (c) 2025 Leland Stanford Junior University
# Copyright (c) 2025 The Regents of the University of California
#
# This file is part of pelicun.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# You should have received a copy of the BSD 3-Clause License along with
# pelicun. If not, see <http://www.opensource.org/licenses/>.
#
# Contributors:
# Adam Zsarnóczay

"""Integration tests for the pelicun DLML (Damage and Loss Model Library) module."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_httpserver.httpserver import HTTPServer

import subprocess  # noqa: S404
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from pelicun.tools import dlml


@pytest.mark.allow_hosts(['localhost', '127.0.0.1'])
def test_full_download_flow_e2e(
    httpserver: HTTPServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify the entire DLML download process using a local HTTP server.

    This test simulates the complete download workflow against a controlled local
    HTTP server that mimics GitHub's API responses and file content delivery.
    """
    # Set up the environment and mock server
    test_owner = 'Test-Owner'
    test_repo = 'Test-Repo'

    monkeypatch.setenv('PELICUN_GITHUB_API_URL', httpserver.url_for('/'))
    monkeypatch.setenv('PELICUN_GITHUB_RAW_URL', httpserver.url_for('/'))
    monkeypatch.setenv('DLML_DATA_DIR', str(tmp_path))
    monkeypatch.setenv('DLML_REPO_OWNER', test_owner)
    monkeypatch.setenv('DLML_REPO_NAME', test_repo)

    # Define fake file content
    model_files_content = 'damage/test_fragility.csv'
    damage_model_content = 'LS_1,LS_2'
    latest_release_payload = {
        'tag_name': 'v2.1.0',
        'target_commitish': 'e2etestingcommit',
    }

    # Build the expected paths for the server
    repo_path = f'/repos/{test_owner}/{test_repo}'
    raw_path = f'/{test_owner}/{test_repo}/e2etestingcommit'

    httpserver.expect_request(f'{repo_path}/releases/latest').respond_with_json(
        latest_release_payload
    )
    httpserver.expect_request(f'{raw_path}/model_files.txt').respond_with_data(
        model_files_content
    )
    httpserver.expect_request(
        f'{raw_path}/damage/test_fragility.csv'
    ).respond_with_data(damage_model_content)

    # Run the top-level function
    dlml.dlml_update(version='latest', use_cache=False)

    # Verify the results on the real filesystem
    manifest_path = tmp_path / 'model_files.txt'
    damage_model_path = tmp_path / 'damage' / 'test_fragility.csv'

    assert manifest_path.exists()
    assert damage_model_path.exists()
    assert manifest_path.read_text() == model_files_content
    assert damage_model_path.read_text() == damage_model_content


# --- File System Interaction Tests ---


@patch('requests.get')
def test_download_file_creates_real_file_in_cache(
    mock_get: MagicMock, tmp_path: Path
) -> None:
    """Verify that the _download_file function properly creates files on disk.

    This test checks that actual filesystem operations work correctly with only
    the network request being mocked.
    """
    # Create test data and paths
    fake_csv_content = b'header1,header2\nvalue1,value2'
    file_url = 'https://example.com/data.csv'
    destination_path = tmp_path / 'data.csv'

    # Mock the network response
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [fake_csv_content]
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    # Call the actual download function with only the network call mocked
    dlml._download_file(file_url, destination_path)

    # Verify file was created and contains correct content
    assert destination_path.exists()
    assert destination_path.read_bytes() == fake_csv_content


@patch('pelicun.tools.dlml.tqdm')
@patch('requests.get')
def test_workflow_with_real_downloads(
    mock_get: MagicMock,
    mock_tqdm: MagicMock,  # noqa: ARG001
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the complete download workflow using real filesystem operations.

    Tests the end-to-end download process with URL-specific mock responses to simulate
    different API responses and file contents.
    """
    monkeypatch.setenv('DLML_DATA_DIR', str(tmp_path))

    # Define the unique content for each file we expect to be downloaded
    model_files_content = 'file1.txt\nfile2.txt'
    file1_content = b'content of file 1'
    file2_content = b'content of file 2'

    # Create URL-specific mock responses
    def mock_get_side_effect(url: str, **kwargs: Any) -> MagicMock:  # noqa: ARG001, ANN401
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None

        if 'releases/tags' in url or 'releases/latest' in url:
            mock_response.json.return_value = {'target_commitish': '1234567'}
        elif 'model_files.txt' in url:
            mock_response.iter_content.return_value = [model_files_content.encode()]
        elif 'file1.txt' in url:
            mock_response.iter_content.return_value = [file1_content]
        elif 'file2.txt' in url:
            mock_response.iter_content.return_value = [file2_content]

        return mock_response

    mock_get.side_effect = mock_get_side_effect

    # Run the download process
    dlml.download_data_files(version='v1.0.0', use_cache=False)

    # Verify expected files were created with correct content
    assert (tmp_path / 'file1.txt').exists()
    assert (tmp_path / 'file2.txt').exists()
    assert (tmp_path / 'file1.txt').read_bytes() == file1_content
    assert (tmp_path / 'file2.txt').read_bytes() == file2_content


@patch('pelicun.tools.dlml.tqdm')
@patch('requests.get')
def test_different_download_modes_integration(
    mock_get: MagicMock,
    mock_tqdm: MagicMock,  # noqa: ARG001
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify all download modes (version, commit, latest) function correctly.

    Tests that the different ways to specify what version to download
    (via version string, commit SHA, or 'latest') all work properly.
    """
    monkeypatch.setenv('DLML_DATA_DIR', str(tmp_path))

    # Define test data
    model_files_content = 'file1.txt\nfile2.txt'
    file_content = b'download modes test content'

    # Create URL-specific mock responses
    def mock_get_side_effect(url: str, **kwargs: Any) -> MagicMock:  # noqa: ARG001, ANN401
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None

        if 'releases/tags' in url or 'releases/latest' in url:
            mock_response.json.return_value = {
                'target_commitish': '1234567',
                'tag_name': 'v1.0.0',
            }
        elif 'git/refs/tags' in url:
            mock_response.json.return_value = {
                'object': {'sha': '1234567', 'type': 'commit'}
            }
        elif 'commits' in url:
            mock_response.json.return_value = [{'sha': '1234567'}]
        elif 'model_files.txt' in url:
            mock_response.iter_content.return_value = [model_files_content.encode()]
        elif 'file1.txt' in url or 'file2.txt' in url:
            mock_response.iter_content.return_value = [file_content]
        else:
            mock_response.iter_content.return_value = [b'fallback content']

        return mock_response

    mock_get.side_effect = mock_get_side_effect

    # Test with version parameter
    dlml.download_data_files(version='v1.0.0', use_cache=False)
    assert (tmp_path / 'file1.txt').exists()
    assert (tmp_path / 'file2.txt').exists()
    assert (tmp_path / 'file1.txt').read_bytes() == file_content

    # Clean up for next test
    (tmp_path / 'file1.txt').unlink()
    (tmp_path / 'file2.txt').unlink()

    # Test with specific commit parameter
    dlml.download_data_files(commit='1234567', use_cache=False)
    assert (tmp_path / 'file1.txt').exists()
    assert (tmp_path / 'file2.txt').exists()
    assert (tmp_path / 'file1.txt').read_bytes() == file_content

    # Clean up for next test
    (tmp_path / 'file1.txt').unlink()
    (tmp_path / 'file2.txt').unlink()

    # Test with latest commit parameter
    dlml.download_data_files(commit='latest', use_cache=False)
    assert (tmp_path / 'file1.txt').exists()
    assert (tmp_path / 'file2.txt').exists()
    assert (tmp_path / 'file1.txt').read_bytes() == file_content


# --- Edge Case Tests ---


@patch('pelicun.tools.dlml.tqdm')
@patch('requests.get')
def test_with_empty_model_files(
    mock_get: MagicMock,
    mock_tqdm: MagicMock,  # noqa: ARG001
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the download process handles empty model_files.txt gracefully.

    Tests that when model_files.txt is empty, the download process completes
    without error and doesn't attempt to download any additional files.
    """
    monkeypatch.setenv('DLML_DATA_DIR', str(tmp_path))

    # Define an empty model_files.txt content
    model_files_content = ''

    def mock_get_side_effect(url: str, **kwargs: Any) -> MagicMock:  # noqa: ARG001, ANN401
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None

        if 'releases/tags' in url or 'releases/latest' in url:
            mock_response.json.return_value = {'target_commitish': '1234567'}
        elif 'model_files.txt' in url:
            mock_response.iter_content.return_value = [model_files_content.encode()]
        else:
            # This should not be reached for empty model files list
            mock_response.iter_content.return_value = [b'file content']

        return mock_response

    mock_get.side_effect = mock_get_side_effect

    dlml.download_data_files(version='v1.0.0', use_cache=False)

    # Verify no additional files were created beyond model_files.txt
    file_count = len(list(tmp_path.glob('**/*')))
    assert file_count == 1  # Just model_files.txt exists


@patch('pelicun.tools.dlml.tqdm')
@patch('requests.get')
def test_with_large_number_of_files(
    mock_get: MagicMock,
    mock_tqdm: MagicMock,  # noqa: ARG001
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the download process works with a large number of files.

    Tests the system's ability to handle downloading many files at once
    by simulating 50 files in the model_files.txt listing.
    """
    monkeypatch.setenv('DLML_DATA_DIR', str(tmp_path))

    # Using a smaller number (50) for test performance
    file_count = 50
    large_file_list = '\n'.join([f'file{i}.txt' for i in range(file_count)])
    file_content = b'file content for large test'

    def mock_get_side_effect(url: str, **kwargs: Any) -> MagicMock:  # noqa: ARG001, ANN401
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None

        if 'releases/tags' in url or 'releases/latest' in url:
            mock_response.json.return_value = {'target_commitish': '1234567'}
        elif 'model_files.txt' in url:
            mock_response.iter_content.return_value = [large_file_list.encode()]
        elif any(f'file{i}.txt' in url for i in range(file_count)):
            mock_response.iter_content.return_value = [file_content]
        else:
            mock_response.iter_content.return_value = [b'fallback content']

        return mock_response

    mock_get.side_effect = mock_get_side_effect

    dlml.download_data_files(version='v1.0.0', use_cache=False)

    # Verify files were created on the filesystem
    assert (
        len(list(tmp_path.glob('*.txt'))) == file_count + 1
    )  # model_files.txt + all files

    # Check content of a sample file
    assert (tmp_path / 'file0.txt').exists()
    assert (tmp_path / 'file0.txt').read_bytes() == file_content


@patch('pelicun.tools.dlml.tqdm')
@patch('requests.get')
def test_with_special_characters_in_paths(
    mock_get: MagicMock,
    mock_tqdm: MagicMock,  # noqa: ARG001
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the download process handles paths with special characters.

    Tests that file paths containing spaces, hash symbols, and question marks
    are correctly processed and the files are created in the expected locations.
    """
    monkeypatch.setenv('DLML_DATA_DIR', str(tmp_path))

    # Special file paths to test
    special_file_list = 'path/with spaces/file.txt\npath/with#hash/file.txt\npath/with&ampersand/file.txt'
    file_content = b'special character file content'

    def mock_get_side_effect(url: str, **kwargs: Any) -> MagicMock:  # noqa: ARG001, ANN401
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None

        if 'releases/tags' in url or 'releases/latest' in url:
            mock_response.json.return_value = {'target_commitish': '1234567'}
        elif 'model_files.txt' in url:
            mock_response.iter_content.return_value = [special_file_list.encode()]
        # URL encoding can vary between platforms/versions, so use a lenient check
        elif 'file.txt' in url and (
            'spaces' in url or 'hash' in url or 'ampersand' in url
        ):
            mock_response.iter_content.return_value = [file_content]
        else:
            mock_response.iter_content.return_value = [b'fallback content']

        return mock_response

    mock_get.side_effect = mock_get_side_effect

    dlml.download_data_files(version='v1.0.0', use_cache=False)

    # Verify the directories and files were created with special characters
    assert (tmp_path / 'path' / 'with spaces' / 'file.txt').exists()
    assert (tmp_path / 'path' / 'with#hash' / 'file.txt').exists()
    assert (tmp_path / 'path' / 'with&ampersand' / 'file.txt').exists()

    # Verify file content
    assert (
        tmp_path / 'path' / 'with spaces' / 'file.txt'
    ).read_bytes() == file_content
    assert (
        tmp_path / 'path' / 'with#hash' / 'file.txt'
    ).read_bytes() == file_content
    assert (
        tmp_path / 'path' / 'with&ampersand' / 'file.txt'
    ).read_bytes() == file_content


@patch('pelicun.tools.dlml.tqdm')
@patch('requests.get')
def test_with_commented_lines_in_model_files(
    mock_get: MagicMock,
    mock_tqdm: MagicMock,  # noqa: ARG001
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that commented lines in model_files.txt are properly ignored.

    Tests that lines starting with # in model_files.txt are skipped during
    download, ensuring only actual file entries are processed.
    """
    monkeypatch.setenv('DLML_DATA_DIR', str(tmp_path))

    # Define model_files.txt with comments
    file_list = """# This is a comment
file1.txt
# This is another comment
file2.txt
# This is a comment with leading whitespace
file3.txt"""
    file_content = b'commented lines test content'

    def mock_get_side_effect(url: str, **kwargs: Any) -> MagicMock:  # noqa: ARG001, ANN401
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None

        if 'releases/tags' in url or 'releases/latest' in url:
            mock_response.json.return_value = {'target_commitish': '1234567'}
        elif 'model_files.txt' in url:
            mock_response.iter_content.return_value = [file_list.encode()]
        elif any(
            file_name in url for file_name in ['file1.txt', 'file2.txt', 'file3.txt']
        ):
            mock_response.iter_content.return_value = [file_content]
        else:
            mock_response.iter_content.return_value = [b'fallback content']

        return mock_response

    mock_get.side_effect = mock_get_side_effect

    dlml.download_data_files(version='v1.0.0', use_cache=False)

    # Check for expected files
    assert (tmp_path / 'file1.txt').exists()
    assert (tmp_path / 'file2.txt').exists()
    assert (tmp_path / 'file3.txt').exists()

    # Verify file content
    assert (tmp_path / 'file1.txt').read_bytes() == file_content
    assert (tmp_path / 'file2.txt').read_bytes() == file_content
    assert (tmp_path / 'file3.txt').read_bytes() == file_content

    # Verify exactly 4 .txt files exist (model_files.txt + 3 data files)
    txt_files = len(list(tmp_path.glob('*.txt')))
    assert (
        txt_files == 4
    ), f"Expected 4 .txt files, got {txt_files}: {list(tmp_path.glob('*.txt'))}"


# --- Integration Tests ---


@patch('requests.get')
@patch('pelicun.tools.dlml.save_cache')
@patch('pelicun.tools.dlml.load_cache')
def test_cache_version_tracking_integration(
    mock_load_cache: MagicMock,
    mock_save_cache: MagicMock,
    mock_get: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that cache version tracking and updating work correctly.

    Tests the interaction between the cache system and version checking,
    ensuring version comparisons are correct and cache gets updated.
    """
    monkeypatch.setenv('DLML_DATA_DIR', str(tmp_path))

    # Create directory to satisfy the pathlib.Path.exists check
    tmp_path.mkdir(parents=True, exist_ok=True)

    # Mock cache with version metadata
    mock_cache = {
        'version': 'v1.0.0',
        'download_type': 'version',
        'commit_sha': '1234567',
        'last_updated': '2025-08-13T10:00:00',
    }
    mock_load_cache.return_value = mock_cache

    # Mock GitHub API response with newer version
    mock_response = MagicMock()
    mock_response.json.return_value = {'tag_name': 'v1.1.0'}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    result = dlml.check_dlml_version()

    # Verify version comparison is correct
    assert result['current_version'] == 'v1.0.0'
    assert result['latest_version'] == 'v1.1.0'
    assert result['update_available'] is True

    # Verify cache is updated
    mock_save_cache.assert_called_once()
    updated_cache = mock_save_cache.call_args[0][1]
    assert updated_cache['current_version'] == 'v1.0.0'
    assert updated_cache['latest_version'] == 'v1.1.0'
    assert updated_cache['update_available'] is True


@patch('pelicun.tools.dlml.check_dlml_data')
def test_import_integration_success(mock_check: MagicMock) -> None:
    """Verify successful import integration with check_dlml_data.

    Tests that the import process works correctly when check_dlml_data succeeds.
    """
    # Mock successful data check
    mock_check.return_value = None

    # Import should succeed without raising exceptions
    try:
        # Simulate the import process
        from pelicun.tools.dlml import check_dlml_data  # noqa: PLC0415

        check_dlml_data()
    except (ImportError, RuntimeError, ValueError) as e:
        pytest.fail(f'Import integration should not raise exception: {e}')


@patch(
    'pelicun.tools.dlml.check_dlml_data', side_effect=RuntimeError('Download failed')
)
def test_import_integration_failure(mock_check_dlml_data: MagicMock) -> None:  # noqa: ARG001
    """Verify that import failures are properly handled.

    Tests that errors during the import process are correctly raised.
    """
    from pelicun.tools.dlml import check_dlml_data  # noqa: PLC0415

    with pytest.raises(RuntimeError, match='Download failed'):
        check_dlml_data()


@patch.object(dlml.logger, 'warning')
@patch('requests.get', side_effect=requests.exceptions.RequestException)
def test_logging_configuration(mock_get: MagicMock, mock_warning: MagicMock) -> None:  # noqa: ARG001
    """Verify that the module-level logger is configured and used correctly.

    Tests that the dlml module's logger is properly configured and that
    logging calls are made at the appropriate points.
    """
    # Verify logger exists and is configured correctly
    assert hasattr(dlml, 'logger')
    assert dlml.logger.name == 'pelicun.dlml'

    # Trigger a warning log by causing a network error
    dlml._get_changed_files('base', 'head', headers={})

    # Verify the warning was logged
    mock_warning.assert_called()


@patch('warnings.warn')
@patch('pelicun.tools.dlml.check_dlml_version')
def test_warning_system_integration(
    mock_check_version: MagicMock,
    mock_warn: MagicMock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify integration with pelicun's warning system.

    Tests that warnings about available updates are correctly issued
    through the pelicun warning system with the proper warning class.
    """
    from pelicun.pelicun_warnings import PelicunWarning  # noqa: PLC0415

    monkeypatch.setenv('DLML_DATA_DIR', str(tmp_path))

    # Create manifest file that will be checked for existence
    manifest_file = tmp_path / 'model_files.txt'
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.touch()

    # Mock version check result with update available
    mock_version_info = {
        'update_available': True,
        'current_version': 'v1.0.0',
        'latest_version': 'v1.1.0',
        'error': None,
    }
    mock_check_version.return_value = mock_version_info

    dlml.check_dlml_data()

    # Verify warning was issued with correct parameters
    mock_warn.assert_called_once()
    args, kwargs = mock_warn.call_args
    assert len(args) >= 2
    assert args[1] == PelicunWarning
    assert 'stacklevel' in kwargs
    assert kwargs['stacklevel'] == 2


# --- Command-Line Interface Integration Tests ---


def test_cli_integration_invalid_action() -> None:
    """Verify CLI correctly handles invalid actions.

    Tests that the CLI returns the expected error code and message
    when an invalid action is provided.
    """
    result = subprocess.run(  # noqa: S603
        ['pelicun', 'dlml', 'invalid'],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )

    # Verify exit code and output
    assert result.returncode == 2  # argparse returns 2 for invalid choice
    assert "invalid choice: 'invalid'" in result.stderr


def test_cli_integration_missing_arguments() -> None:
    """Verify CLI correctly handles missing required arguments.

    Tests that the CLI returns the expected error code and message
    when required arguments are missing.
    """
    result = subprocess.run(  # noqa: S603
        ['pelicun', 'dlml'],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )

    # Verify exit code and output
    assert result.returncode == 2  # argparse returns 2 for missing required argument
    assert 'required' in result.stderr

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

"""These are unit tests on the dlml module of pelicun."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import requests

from pelicun.tools import dlml

# --- Commit SHA Validation Tests ---


def test_validate_commit_sha_valid() -> None:
    """Test validation of valid 7-character commit SHA."""
    # Test valid 7-character hexadecimal SHA
    assert dlml.validate_commit_sha('1234567') is True
    assert dlml.validate_commit_sha('abcdef0') is True
    assert dlml.validate_commit_sha('7890ABC') is True  # Mixed case


def test_validate_commit_sha_invalid() -> None:
    """Test validation of invalid commit SHA."""
    # Test invalid SHAs
    assert dlml.validate_commit_sha('123456') is False  # Too short
    assert dlml.validate_commit_sha('12345678') is False  # Too long
    assert dlml.validate_commit_sha('123xyz!') is False  # Invalid characters


def test_validate_commit_sha_latest() -> None:
    """Test validation of the special 'latest' value."""
    assert dlml.validate_commit_sha('latest') is True


# --- File Hash Calculation Tests ---


def test_get_file_hash_existing_file() -> None:
    """Test hash calculation for existing files."""
    # Create a temporary file with known content
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(b'test content')
        temp_file_path = temp_file.name

    try:
        # Calculate hash
        file_hash = dlml.get_file_hash(temp_file_path)

        # Verify hash is not None and is a string
        assert file_hash is not None
        assert isinstance(file_hash, str)
        assert len(file_hash) == 32  # MD5 hash is 32 characters
    finally:
        # Clean up
        os.unlink(temp_file_path)


def test_get_file_hash_nonexistent_file() -> None:
    """Test hash calculation for non-existent files."""
    # Use a path that definitely doesn't exist
    nonexistent_path = '/path/that/definitely/does/not/exist'

    # Calculate hash
    file_hash = dlml.get_file_hash(nonexistent_path)

    # Verify hash is None
    assert file_hash is None


def test_get_file_hash_different_content() -> None:
    """Test hash calculation for files with different content."""
    # Create two temporary files with different content
    with tempfile.NamedTemporaryFile(delete=False) as temp_file1:
        temp_file1.write(b'content 1')
        temp_file1_path = temp_file1.name

    with tempfile.NamedTemporaryFile(delete=False) as temp_file2:
        temp_file2.write(b'content 2')
        temp_file2_path = temp_file2.name

    try:
        # Calculate hashes
        hash1 = dlml.get_file_hash(temp_file1_path)
        hash2 = dlml.get_file_hash(temp_file2_path)

        # Verify hashes are different
        assert hash1 != hash2
    finally:
        # Clean up
        os.unlink(temp_file1_path)
        os.unlink(temp_file2_path)


def test_get_file_hash_empty_file() -> None:
    """Test hash calculation for empty files."""
    # Create a temporary empty file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_path = temp_file.name

    try:
        # Calculate hash
        file_hash = dlml.get_file_hash(temp_file_path)

        # Verify hash is not None and is a string
        assert file_hash is not None
        assert isinstance(file_hash, str)
        assert len(file_hash) == 32  # MD5 hash is 32 characters

        # Known MD5 hash for empty file
        assert file_hash == 'd41d8cd98f00b204e9800998ecf8427e'
    finally:
        # Clean up
        os.unlink(temp_file_path)


# --- Cache Management Tests ---


def test_load_cache_valid_file() -> None:
    """Test loading from a valid cache file."""
    # Create a mock cache file
    cache_data = {'commit_sha': '1234567', 'files': {'file1.txt': 'hash1'}}
    mock_file = mock_open(read_data=json.dumps(cache_data))

    # Mock os.path.exists to return True
    with patch('os.path.exists', return_value=True), patch(
        'builtins.open', mock_file
    ):
        # Load cache
        loaded_cache = dlml.load_cache('cache_file.json')

        # Verify cache is loaded correctly
        assert loaded_cache == cache_data


def test_load_cache_nonexistent_file() -> None:
    """Test loading from a non-existent cache file."""
    # Mock os.path.exists to return False
    with patch('os.path.exists', return_value=False):
        # Load cache
        loaded_cache = dlml.load_cache('nonexistent_cache_file.json')

        # Verify empty cache is returned
        assert loaded_cache == {}


def test_load_cache_corrupted_file() -> None:
    """Test loading from a corrupted cache file."""
    # Create a mock corrupted cache file
    mock_file = mock_open(read_data='invalid json')

    # Mock os.path.exists to return True
    with patch('os.path.exists', return_value=True), patch(
        'builtins.open', mock_file
    ):
        # Load cache
        loaded_cache = dlml.load_cache('corrupted_cache_file.json')

        # Verify empty cache is returned
        assert loaded_cache == {}


def test_save_cache() -> None:
    """Test saving cache data to a file."""
    # Create mock cache data
    cache_data = {'commit_sha': '1234567', 'files': {'file1.txt': 'hash1'}}

    # Mock open and os.makedirs
    mock_file = mock_open()

    with patch('os.makedirs') as mock_makedirs, patch('builtins.open', mock_file):
        # Save cache
        dlml.save_cache('cache_dir/cache_file.json', cache_data)

        # Verify os.makedirs was called with the correct arguments
        mock_makedirs.assert_called_once_with('cache_dir', exist_ok=True)

        # Verify open was called with the correct arguments
        mock_file.assert_called_once_with('cache_dir/cache_file.json', 'w')

        # Verify json.dump was called with the correct arguments
        handle = mock_file()
        # json.dump writes in multiple chunks, so just verify write was called
        assert handle.write.call_count > 0


# --- File Download Tests ---


def test_download_file_success() -> None:
    """Test downloading a file from a valid URL."""
    # Create a mock response
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b'chunk1', b'chunk2']

    # Mock requests.get to return the mock response
    with patch('requests.get', return_value=mock_response) as mock_get, patch(
        'os.makedirs'
    ) as mock_makedirs, patch('builtins.open', mock_open()) as mock_file:
        # Download file
        dlml._download_file('http://example.com/file.txt', 'dir/file.txt')

        # Verify requests.get was called with the correct arguments
        mock_get.assert_called_once_with('http://example.com/file.txt', stream=True)

        # Verify os.makedirs was called with the correct arguments
        mock_makedirs.assert_called_once_with('dir', exist_ok=True)

        # Verify open was called with the correct arguments
        mock_file.assert_called_once_with('dir/file.txt', 'wb')

        # Verify write was called for each chunk
        handle = mock_file()
        assert handle.write.call_count == 2
        handle.write.assert_any_call(b'chunk1')
        handle.write.assert_any_call(b'chunk2')


def test_download_file_request_exception() -> None:
    """Test handling of download failures."""
    # Mock requests.get to raise an exception
    with patch(
        'requests.get', side_effect=requests.exceptions.RequestException('Error')
    ), patch('os.makedirs'):
        # Download file and verify exception is raised
        with pytest.raises(RuntimeError) as excinfo:
            dlml._download_file('http://example.com/file.txt', 'dir/file.txt')

        # Verify error message
        assert 'Failed to download file: http://example.com/file.txt' in str(
            excinfo.value
        )


# --- DLML Download Function Tests ---


def test_download_data_files_with_version() -> None:
    """Test downloading with a specific version."""
    # Mock responses
    mock_release_response = MagicMock()
    mock_release_response.json.return_value = {'target_commitish': '1234567'}

    # Mock functions
    with patch(
        'requests.get', return_value=mock_release_response
    ) as mock_get, patch(
        'pelicun.tools.dlml._download_file'
    ) as mock_download, patch(
        'pelicun.tools.dlml.load_cache', return_value={}
    ), patch('pelicun.tools.dlml.save_cache'), patch(
        'pelicun.tools.dlml.get_file_hash', return_value='hash'
    ), patch('os.path.dirname', return_value='/path'), patch(
        'os.path.join', return_value='/path/to/file'
    ), patch('os.makedirs'), patch(
        'builtins.open', mock_open(read_data='file1.txt\nfile2.txt')
    ), patch('pelicun.tools.dlml.tqdm'):
        # Download data files
        dlml.download_data_files(version='v1.0.0')

        # Verify requests.get was called with the correct arguments
        mock_get.assert_called_once()
        assert 'releases/tags/v1.0.0' in mock_get.call_args[0][0]

        # Verify _download_file was called for model_files.txt and each file in it
        assert mock_download.call_count == 3  # model_files.txt + 2 files


def test_download_data_files_with_commit() -> None:
    """Test downloading with a specific commit."""
    # Mock functions
    with patch('pelicun.tools.dlml.validate_commit_sha', return_value=True), patch(
        'pelicun.tools.dlml._download_file'
    ) as mock_download, patch(
        'pelicun.tools.dlml.load_cache', return_value={}
    ), patch('pelicun.tools.dlml.save_cache'), patch(
        'pelicun.tools.dlml.get_file_hash', return_value='hash'
    ), patch('os.path.dirname', return_value='/path'), patch(
        'os.path.join', return_value='/path/to/file'
    ), patch('os.makedirs'), patch(
        'builtins.open', mock_open(read_data='file1.txt\nfile2.txt')
    ), patch('pelicun.tools.dlml.tqdm'):
        # Download data files
        dlml.download_data_files(commit='1234567')

        # Verify _download_file was called for model_files.txt and each file in it
        assert mock_download.call_count == 3  # model_files.txt + 2 files


def test_download_data_files_with_latest_commit() -> None:
    """Test downloading with the 'latest' commit."""
    # Mock responses
    mock_commits_response = MagicMock()
    mock_commits_response.json.return_value = [{'sha': '1234567'}]

    # Mock functions
    with patch(
        'requests.get', return_value=mock_commits_response
    ) as mock_get, patch(
        'pelicun.tools.dlml._download_file'
    ) as mock_download, patch(
        'pelicun.tools.dlml.load_cache', return_value={}
    ), patch('pelicun.tools.dlml.save_cache'), patch(
        'pelicun.tools.dlml.get_file_hash', return_value='hash'
    ), patch('os.path.dirname', return_value='/path'), patch(
        'os.path.join', return_value='/path/to/file'
    ), patch('os.makedirs'), patch(
        'builtins.open', mock_open(read_data='file1.txt\nfile2.txt')
    ), patch('pelicun.tools.dlml.tqdm'):
        # Download data files
        dlml.download_data_files(commit='latest')

        # Verify requests.get was called with the correct arguments
        mock_get.assert_called_once()
        assert 'commits' in mock_get.call_args[0][0]

        # Verify _download_file was called for model_files.txt and each file in it
        assert mock_download.call_count == 3  # model_files.txt + 2 files


def test_download_data_files_with_cache() -> None:
    """Test downloading with caching enabled."""
    # Mock cache with existing files
    mock_cache = {
        'commit_sha': '1234567',
        'files': {'file1.txt': 'hash1', 'file2.txt': 'hash2'},
    }

    # Mock functions
    with patch('pelicun.tools.dlml.validate_commit_sha', return_value=True), patch(
        'pelicun.tools.dlml._download_file'
    ) as mock_download, patch(
        'pelicun.tools.dlml.load_cache', return_value=mock_cache
    ), patch('pelicun.tools.dlml.save_cache'), patch(
        'pelicun.tools.dlml.get_file_hash', return_value='hash1'
    ), patch('os.path.dirname', return_value='/path'), patch(
        'os.path.join', return_value='/path/to/file'
    ), patch('os.makedirs'), patch(
        'builtins.open', mock_open(read_data='file1.txt\nfile2.txt')
    ), patch('pelicun.tools.dlml.tqdm'):
        # Download data files with the same commit as in cache
        dlml.download_data_files(commit='1234567')

        # Verify _download_file was not called since files are in cache
        assert mock_download.call_count == 0


def test_download_data_files_invalid_commit() -> None:
    """Test handling of invalid versions or commits."""
    # Mock functions
    with patch('pelicun.tools.dlml.validate_commit_sha', return_value=False):
        # Download data files with invalid commit and verify exception is raised
        with pytest.raises(ValueError) as excinfo:
            dlml.download_data_files(commit='invalid')

        # Verify error message
        assert 'Invalid commit SHA format' in str(excinfo.value)


def test_download_data_files_network_error() -> None:
    """Test handling of network errors during API calls."""
    # Mock requests.get to raise an exception
    with patch(
        'requests.get', side_effect=requests.exceptions.RequestException('Error')
    ), patch('os.path.dirname', return_value='/path'), patch(
        'os.path.join', return_value='/path/to/file'
    ), patch('os.makedirs'):
        # Download data files and verify exception is raised
        with pytest.raises(RuntimeError) as excinfo:
            dlml.download_data_files(version='v1.0.0')

        # Verify error message
        assert 'Failed to fetch release info' in str(excinfo.value)


# --- Error Handling Tests ---


def test_download_data_files_empty_repository() -> None:
    """Test handling when GitHub API returns empty commits list."""
    # Mock responses - empty commits list
    mock_commits_response = MagicMock()
    mock_commits_response.json.return_value = []  # Empty commits list

    # Mock functions
    with patch('requests.get', return_value=mock_commits_response), patch(
        'os.path.dirname', return_value='/path'
    ), patch('os.path.join', return_value='/path/to/file'), patch('os.makedirs'):
        # Download data files with latest commit and verify exception is raised
        with pytest.raises(RuntimeError) as excinfo:
            dlml.download_data_files(commit='latest')

        # Verify error message
        assert 'No commits found in the repository' in str(excinfo.value)


def test_download_data_files_missing_commit_sha_in_release() -> None:
    """Test handling when release data doesn't contain target_commitish."""
    # Mock responses - release data without target_commitish
    mock_release_response = MagicMock()
    mock_release_response.json.return_value = {
        'tag_name': 'v1.0.0'
    }  # Missing target_commitish

    # Mock functions
    with patch('requests.get', return_value=mock_release_response), patch(
        'os.path.dirname', return_value='/path'
    ), patch('os.path.join', return_value='/path/to/file'), patch('os.makedirs'):
        # Download data files and verify exception is raised
        with pytest.raises(RuntimeError) as excinfo:
            dlml.download_data_files(version='v1.0.0')

        # Verify error message
        assert "Could not find commit SHA for release 'v1.0.0'" in str(excinfo.value)


def test_download_data_files_model_file_not_found() -> None:
    """Test handling when model_files.txt doesn't exist after download."""
    # Mock responses
    mock_release_response = MagicMock()
    mock_release_response.json.return_value = {'target_commitish': '1234567'}

    # Mock functions - simulate FileNotFoundError when opening model_files.txt
    with patch('requests.get', return_value=mock_release_response), patch(
        'pelicun.tools.dlml._download_file'
    ), patch('pelicun.tools.dlml.load_cache', return_value={}), patch(
        'os.path.dirname', return_value='/path'
    ), patch('os.path.join', return_value='/path/to/file'), patch(
        'os.makedirs'
    ), patch('builtins.open', side_effect=FileNotFoundError('File not found')):
        # Download data files and verify exception is raised
        with pytest.raises(RuntimeError) as excinfo:
            dlml.download_data_files(version='v1.0.0')

        # Verify error message
        assert 'Model file list not found at' in str(excinfo.value)


def test_download_data_files_model_file_read_error() -> None:
    """Test handling of general file reading exceptions."""
    # Mock responses
    mock_release_response = MagicMock()
    mock_release_response.json.return_value = {'target_commitish': '1234567'}

    # Mock functions - simulate general Exception when opening model_files.txt
    with patch('requests.get', return_value=mock_release_response), patch(
        'pelicun.tools.dlml._download_file'
    ), patch('pelicun.tools.dlml.load_cache', return_value={}), patch(
        'os.path.dirname', return_value='/path'
    ), patch('os.path.join', return_value='/path/to/file'), patch(
        'os.makedirs'
    ), patch('builtins.open', side_effect=Exception('General read error')):
        # Download data files and verify exception is raised
        with pytest.raises(RuntimeError) as excinfo:
            dlml.download_data_files(version='v1.0.0')

        # Verify error message
        assert 'Error reading model file list' in str(excinfo.value)


# --- Progress Reporting Tests ---


def test_progress_bar_initialization() -> None:
    """Test progress bar initialization with different file counts."""
    # Mock tqdm
    with patch('pelicun.tools.dlml.tqdm') as mock_tqdm, patch(
        'pelicun.tools.dlml._download_file'
    ), patch('pelicun.tools.dlml.load_cache', return_value={}), patch(
        'pelicun.tools.dlml.save_cache'
    ), patch('pelicun.tools.dlml.get_file_hash', return_value='hash'), patch(
        'os.path.dirname', return_value='/path'
    ), patch('os.path.join', return_value='/path/to/file'), patch(
        'os.makedirs'
    ), patch('requests.get') as mock_get, patch(
        'builtins.open', mock_open(read_data='file1.txt\nfile2.txt')
    ):
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {'target_commitish': '1234567'}
        mock_get.return_value = mock_response

        # Download data files
        dlml.download_data_files(version='v1.0.0', use_cache=False)

        # Verify tqdm was called with the correct arguments
        mock_tqdm.assert_called_once()
        _args, kwargs = mock_tqdm.call_args
        assert kwargs['total'] == 2  # 2 files in model_files.txt
        assert kwargs['desc'] == 'Downloading files'
        assert kwargs['unit'] == 'file'


def test_progress_bar_updates() -> None:
    """Test progress bar updates during downloads."""
    # Mock tqdm as a context manager
    mock_progress = MagicMock()
    mock_tqdm = MagicMock()
    mock_tqdm.return_value.__enter__.return_value = mock_progress

    with patch('pelicun.tools.dlml.tqdm', mock_tqdm), patch(
        'pelicun.tools.dlml._download_file'
    ), patch('pelicun.tools.dlml.load_cache', return_value={}), patch(
        'pelicun.tools.dlml.save_cache'
    ), patch('pelicun.tools.dlml.get_file_hash', return_value='hash'), patch(
        'os.path.dirname', return_value='/path'
    ), patch('os.path.join', return_value='/path/to/file'), patch(
        'os.makedirs'
    ), patch('requests.get') as mock_get, patch(
        'builtins.open', mock_open(read_data='file1.txt\nfile2.txt')
    ):
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {'target_commitish': '1234567'}
        mock_get.return_value = mock_response

        # Download data files
        dlml.download_data_files(version='v1.0.0', use_cache=False)

        # Verify progress bar was updated for each file
        assert mock_progress.update.call_count == 2  # Called once for each file
        assert (
            mock_progress.set_description.call_count >= 2
        )  # Called at least once for each file


def test_progress_reporting_for_skipped_files() -> None:
    """Test progress reporting for skipped files."""
    # Mock cache with existing files but different commit SHA to avoid early return
    mock_cache = {
        'commit_sha': 'abcdefg',  # Different commit SHA
        'files': {'file1.txt': 'hash1', 'file2.txt': 'hash2'},
    }

    # Mock tqdm as a context manager
    mock_progress = MagicMock()
    mock_tqdm = MagicMock()
    mock_tqdm.return_value.__enter__.return_value = mock_progress

    # Mock get_file_hash to return matching hashes for files to simulate caching
    def mock_get_file_hash_side_effect(path):
        # Return matching hashes for both files so they get skipped
        if path.endswith('file1.txt'):
            return 'hash1'  # Matches cache, will be skipped
        elif path.endswith('file2.txt'):
            return 'hash2'  # Matches cache, will be skipped
        return None  # For non-existent files

    # Mock os.path.join to return predictable paths
    def mock_path_join(*args):
        return '/'.join(args)

    with patch('pelicun.tools.dlml.tqdm', mock_tqdm), patch(
        'pelicun.tools.dlml._download_file'
    ) as mock_download, patch(
        'pelicun.tools.dlml.load_cache', return_value=mock_cache
    ), patch('pelicun.tools.dlml.save_cache'), patch(
        'pelicun.tools.dlml.get_file_hash',
        side_effect=mock_get_file_hash_side_effect,
    ), patch('os.path.dirname', return_value='/path'), patch(
        'os.path.join', side_effect=mock_path_join
    ), patch('os.makedirs'), patch('os.path.exists', return_value=True), patch(
        'requests.get'
    ) as mock_get, patch(
        'builtins.open', mock_open(read_data='file1.txt\nfile2.txt')
    ):
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {'target_commitish': '1234567'}
        mock_get.return_value = mock_response

        # Download data files
        dlml.download_data_files(commit='1234567', use_cache=True)

        # Verify progress bar was updated for each file, even though they were skipped
        assert mock_progress.update.call_count == 2  # Called once for each file
        assert mock_download.call_count == 1  # Only model_files.txt was downloaded


# --- Command-Line Interface Tests ---


# Integration tests that verify the actual CLI interface works
def test_cli_integration_invalid_action() -> None:
    """Integration test: CLI with invalid action (not 'update')."""
    # Run the CLI with invalid action
    result = subprocess.run(
        ['pelicun', 'dlml', 'invalid'], capture_output=True, text=True
    )

    # Verify exit code and output
    assert result.returncode == 2  # argparse returns 2 for invalid choice
    assert "invalid choice: 'invalid'" in result.stderr


def test_cli_integration_missing_arguments() -> None:
    """Integration test: CLI with insufficient arguments."""
    # Run the CLI with no arguments for dlml
    result = subprocess.run(['pelicun', 'dlml'], capture_output=True, text=True)

    # Verify exit code and output
    assert result.returncode == 2  # argparse returns 2 for missing required argument
    assert 'required' in result.stderr


# Unit tests for the main() function


def test_main_version_download() -> None:
    """Test successful version-based download via dlml_update() function."""
    with patch('pelicun.tools.dlml.download_data_files') as mock_download:
        # Test dlml_update function with version argument
        dlml.dlml_update(version='v1.0.0', use_cache=True)

        # Verify successful execution
        mock_download.assert_called_once_with(version='v1.0.0', use_cache=True)


def test_main_commit_download() -> None:
    """Test successful commit-based download via dlml_update() function."""
    with patch('pelicun.tools.dlml.download_data_files') as mock_download:
        # Test dlml_update function with commit argument
        dlml.dlml_update(commit='1234567', use_cache=True)

        # Verify successful execution
        mock_download.assert_called_once_with(commit='1234567', use_cache=True)


def test_main_no_cache_flag() -> None:
    """Test no-cache functionality via dlml_update() function."""
    with patch('pelicun.tools.dlml.download_data_files') as mock_download:
        # Test dlml_update function with use_cache=False
        dlml.dlml_update(version='v1.0.0', use_cache=False)

        # Verify successful execution
        mock_download.assert_called_once_with(version='v1.0.0', use_cache=False)


def test_main_download_error_handling() -> None:
    """Test error handling when download_data_files raises exceptions via dlml_update() function."""
    with patch(
        'pelicun.tools.dlml.download_data_files',
        side_effect=RuntimeError('Test error'),
    ), pytest.raises(RuntimeError, match='Data download failed: Test error'):
        # Test dlml_update function with mocked exception
        dlml.dlml_update(version='v1.0.0', use_cache=True)


def test_main_commit_with_no_cache() -> None:
    """Test dlml_update() function with commit and no cache."""
    with patch('pelicun.tools.dlml.download_data_files') as mock_download:
        # Test dlml_update function with commit and use_cache=False
        dlml.dlml_update(commit='1234567', use_cache=False)

        # Verify successful execution
        mock_download.assert_called_once_with(commit='1234567', use_cache=False)


def test_main_default_latest_version() -> None:
    """Test dlml_update() function with default 'latest' version when no version specified."""
    with patch('pelicun.tools.dlml.download_data_files') as mock_download:
        # Test dlml_update function with default latest version
        dlml.dlml_update()

        # Verify successful execution
        mock_download.assert_called_once_with(version='latest', use_cache=True)


# --- Enhanced DLML Features Tests ---


def test_check_dlml_version_with_cached_result() -> None:
    """Test check_dlml_version returns cached result when within 24 hours."""
    from datetime import datetime, timedelta

    # Mock cache data with recent version check
    recent_time = datetime.now() - timedelta(hours=12)  # noqa: DTZ005
    mock_cache = {
        'last_version_check': recent_time.isoformat(),
        'update_available': True,
        'current_version': 'v1.0.0',
        'latest_version': 'v1.1.0',
        'version_check_error': None,
    }

    with patch('pelicun.tools.dlml.load_cache', return_value=mock_cache), patch(
        'pelicun.tools.dlml.save_cache'
    ), patch('os.path.dirname', return_value='/path'), patch(
        'os.path.join', return_value='/path/cache.json'
    ), patch('os.path.exists', return_value=True):
        result = dlml.check_dlml_version()

        # Should return cached result without making API call
        assert result['update_available'] is True
        assert result['current_version'] == 'v1.0.0'
        assert result['latest_version'] == 'v1.1.0'
        assert result['last_check'] == recent_time.isoformat()
        assert result['error'] is None


def test_check_dlml_version_with_expired_cache() -> None:
    """Test check_dlml_version performs new check when cache is expired."""
    from datetime import datetime, timedelta

    # Mock cache data with old version check
    old_time = datetime.now() - timedelta(days=2)  # noqa: DTZ005
    mock_cache = {'last_version_check': old_time.isoformat(), 'version': 'v1.0.0'}

    # Mock GitHub API response
    mock_response = MagicMock()
    mock_response.json.return_value = {'tag_name': 'v1.1.0'}
    mock_response.raise_for_status.return_value = None

    with patch('pelicun.tools.dlml.load_cache', return_value=mock_cache), patch(
        'pelicun.tools.dlml.save_cache'
    ) as mock_save, patch('requests.get', return_value=mock_response), patch(
        'os.path.dirname', return_value='/path'
    ), patch('os.path.join', return_value='/path/cache.json'), patch(
        'os.path.exists', return_value=True
    ):
        result = dlml.check_dlml_version()

        # Should perform new version check
        assert result['update_available'] is True
        assert result['current_version'] == 'v1.0.0'
        assert result['latest_version'] == 'v1.1.0'
        assert result['error'] is None

        # Should save updated cache
        mock_save.assert_called_once()


def test_check_dlml_version_with_network_error() -> None:
    """Test check_dlml_version handles network errors gracefully."""
    mock_cache = {'version': 'v1.0.0'}

    with patch('pelicun.tools.dlml.load_cache', return_value=mock_cache), patch(
        'pelicun.tools.dlml.save_cache'
    ), patch(
        'requests.get',
        side_effect=requests.exceptions.RequestException('Network error'),
    ), patch('os.path.dirname', return_value='/path'), patch(
        'os.path.join', return_value='/path/cache.json'
    ), patch('os.path.exists', return_value=True):
        result = dlml.check_dlml_version()

        # Should handle error gracefully
        assert result['update_available'] is False
        assert result['current_version'] == 'v1.0.0'
        assert result['latest_version'] is None
        assert 'Failed to check for updates' in result['error']


def test_check_dlml_version_semantic_version_comparison() -> None:
    """Test check_dlml_version correctly compares semantic versions."""
    mock_cache = {'version': 'v1.2.0'}

    # Mock GitHub API response with newer version
    mock_response = MagicMock()
    mock_response.json.return_value = {'tag_name': 'v1.3.0'}
    mock_response.raise_for_status.return_value = None

    with patch('pelicun.tools.dlml.load_cache', return_value=mock_cache), patch(
        'pelicun.tools.dlml.save_cache'
    ), patch('requests.get', return_value=mock_response), patch(
        'os.path.dirname', return_value='/path'
    ), patch('os.path.join', return_value='/path/cache.json'), patch(
        'os.path.exists', return_value=True
    ):
        result = dlml.check_dlml_version()

        # Should detect update is available
        assert result['update_available'] is True
        assert result['current_version'] == 'v1.2.0'
        assert result['latest_version'] == 'v1.3.0'


def test_check_dlml_version_with_commit_based_version() -> None:
    """Test check_dlml_version handles commit-based versions."""
    mock_cache = {'version': 'commit-abc1234'}

    # Mock GitHub API response
    mock_response = MagicMock()
    mock_response.json.return_value = {'tag_name': 'v1.1.0'}
    mock_response.raise_for_status.return_value = None

    with patch('pelicun.tools.dlml.load_cache', return_value=mock_cache), patch(
        'pelicun.tools.dlml.save_cache'
    ), patch('requests.get', return_value=mock_response), patch(
        'os.path.dirname', return_value='/path'
    ), patch('os.path.join', return_value='/path/cache.json'), patch(
        'os.path.exists', return_value=True
    ):
        result = dlml.check_dlml_version()

        # Should always suggest update for commit-based versions
        assert result['update_available'] is True
        assert result['current_version'] == 'commit-abc1234'
        assert result['latest_version'] == 'v1.1.0'


def test_check_dlml_data_with_missing_data() -> None:
    """Test check_dlml_data downloads data when missing."""
    with patch('os.path.exists', return_value=False), patch(
        'pelicun.tools.dlml.download_data_files'
    ) as mock_download, patch('pelicun.tools.dlml.logger') as mock_logger, patch(
        'os.path.dirname', return_value='/path'
    ), patch('os.path.join', return_value='/path/dlml'):
        dlml.check_dlml_data()

        # Should attempt to download data
        mock_download.assert_called_once_with(version='latest', use_cache=True)

        # Should log appropriate messages
        assert mock_logger.info.call_count >= 2


def test_check_dlml_data_with_existing_data_no_update() -> None:
    """Test check_dlml_data with existing data and no update available."""
    # Mock version check result - no update available
    mock_version_info = {
        'update_available': False,
        'current_version': 'v1.0.0',
        'latest_version': 'v1.0.0',
        'error': None,
    }

    with patch('os.path.exists', return_value=True), patch(
        'os.path.isdir', return_value=True
    ), patch('os.listdir', return_value=['model1.json']), patch(
        'pelicun.tools.dlml.check_dlml_version', return_value=mock_version_info
    ), patch('os.path.dirname', return_value='/path'), patch(
        'os.path.join', return_value='/path/dlml'
    ):
        # Should not raise any warnings or exceptions
        dlml.check_dlml_data()


def test_check_dlml_data_with_existing_data_update_available() -> None:
    """Test check_dlml_data with existing data and update available."""
    # Mock version check result - update available
    mock_version_info = {
        'update_available': True,
        'current_version': 'v1.0.0',
        'latest_version': 'v1.1.0',
        'error': None,
    }

    with patch('os.path.exists', return_value=True), patch(
        'os.path.isdir', return_value=True
    ), patch('os.listdir', return_value=['model1.json']), patch(
        'pelicun.tools.dlml.check_dlml_version', return_value=mock_version_info
    ), patch('warnings.warn') as mock_warn, patch(
        'os.path.dirname', return_value='/path'
    ), patch('os.path.join', return_value='/path/dlml'):
        dlml.check_dlml_data()

        # Should issue warning about update availability
        mock_warn.assert_called_once()
        warning_msg = mock_warn.call_args[0][0]
        assert 'DLML data update available' in warning_msg
        assert 'v1.0.0' in warning_msg
        assert 'v1.1.0' in warning_msg
        assert 'pelicun dlml update' in warning_msg


def test_check_dlml_data_download_failure() -> None:
    """Test check_dlml_data handles download failures appropriately."""
    with patch('os.path.exists', return_value=False), patch(
        'pelicun.tools.dlml.download_data_files',
        side_effect=requests.exceptions.ConnectionError('Network error'),
    ), patch('os.path.dirname', return_value='/path'), patch(
        'os.path.join', return_value='/path/dlml'
    ):
        # Should raise RuntimeError with detailed message
        with pytest.raises(RuntimeError) as exc_info:
            dlml.check_dlml_data()

        error_msg = str(exc_info.value)
        assert 'Network error while downloading DLML data' in error_msg
        assert 'pelicun dlml update' in error_msg


def test_check_dlml_data_permission_error() -> None:
    """Test check_dlml_data handles permission errors appropriately."""
    with patch('os.path.exists', return_value=False), patch(
        'pelicun.tools.dlml.download_data_files',
        side_effect=PermissionError('Access denied'),
    ), patch('os.path.dirname', return_value='/path'), patch(
        'os.path.join', return_value='/path/dlml'
    ):
        # Should raise RuntimeError with permission-specific message
        with pytest.raises(RuntimeError) as exc_info:
            dlml.check_dlml_data()

        error_msg = str(exc_info.value)
        assert 'Permission error while downloading DLML data' in error_msg
        assert 'permissions' in error_msg


def test_check_dlml_data_version_check_failure() -> None:
    """Test check_dlml_data handles version check failures gracefully."""
    with patch('os.path.exists', return_value=True), patch(
        'os.path.isdir', return_value=True
    ), patch('os.listdir', return_value=['model1.json']), patch(
        'pelicun.tools.dlml.check_dlml_version',
        side_effect=Exception('Version check failed'),
    ), patch('pelicun.tools.dlml.logger') as mock_logger, patch(
        'os.path.dirname', return_value='/path'
    ), patch('os.path.join', return_value='/path/dlml'):
        # Should not raise exception, just log debug message
        dlml.check_dlml_data()

        # Should log debug message about version check failure
        mock_logger.debug.assert_called_once()
        debug_msg = mock_logger.debug.call_args[0][0]
        assert 'Version check failed' in debug_msg


def test_enhanced_cache_with_version_metadata() -> None:
    """Test that download_data_files stores version metadata in cache."""
    mock_cache = {}

    # Mock GitHub API response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        'target_commitish': '1234567',
        'tag_name': 'v1.2.0',
    }
    mock_response.raise_for_status.return_value = None

    with patch('pelicun.tools.dlml.load_cache', return_value=mock_cache), patch(
        'pelicun.tools.dlml.save_cache'
    ) as mock_save, patch('pelicun.tools.dlml._download_file'), patch(
        'pelicun.tools.dlml.get_file_hash', return_value='hash123'
    ), patch('requests.get', return_value=mock_response), patch(
        'os.path.dirname', return_value='/path'
    ), patch('os.path.join', side_effect=lambda *args: '/'.join(args)), patch(
        'os.makedirs'
    ), patch('os.path.exists', return_value=True), patch(
        'builtins.open', mock_open(read_data='model1.json')
    ):
        # Test version-based download
        dlml.download_data_files(version='v1.2.0', use_cache=True)

        # Verify cache was saved with version metadata
        mock_save.assert_called_once()
        saved_cache = mock_save.call_args[0][1]

        assert saved_cache['version'] == 'v1.2.0'
        assert saved_cache['download_type'] == 'version'
        assert saved_cache['commit_sha'] == '1234567'
        assert 'last_updated' in saved_cache
        assert 'files' in saved_cache


def test_enhanced_cache_with_commit_metadata() -> None:
    """Test that download_data_files stores commit metadata in cache."""
    mock_cache = {}

    # Mock GitHub API response for commits
    mock_response = MagicMock()
    mock_response.json.return_value = [{'sha': 'abcdef1234567890'}]
    mock_response.raise_for_status.return_value = None

    with patch('pelicun.tools.dlml.load_cache', return_value=mock_cache), patch(
        'pelicun.tools.dlml.save_cache'
    ) as mock_save, patch('pelicun.tools.dlml._download_file'), patch(
        'pelicun.tools.dlml.get_file_hash', return_value='hash123'
    ), patch('requests.get', return_value=mock_response), patch(
        'os.path.dirname', return_value='/path'
    ), patch('os.path.join', side_effect=lambda *args: '/'.join(args)), patch(
        'os.makedirs'
    ), patch('os.path.exists', return_value=True), patch(
        'builtins.open', mock_open(read_data='model1.json')
    ):
        # Test commit-based download
        dlml.download_data_files(commit='latest', use_cache=True)

        # Verify cache was saved with commit metadata
        mock_save.assert_called_once()
        saved_cache = mock_save.call_args[0][1]

        assert saved_cache['version'] == 'commit-abcdef1'
        assert saved_cache['download_type'] == 'commit'
        assert saved_cache['commit_sha'] == 'abcdef1234567890'

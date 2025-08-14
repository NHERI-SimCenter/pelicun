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

import os
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest
import requests

from pelicun.tools import dlml


# --- Commit SHA Validation Tests ---

def test_validate_commit_sha_valid():
    """Test validation of valid 7-character commit SHA."""
    # Test valid 7-character hexadecimal SHA
    assert dlml.validate_commit_sha("1234567") is True
    assert dlml.validate_commit_sha("abcdef0") is True
    assert dlml.validate_commit_sha("7890ABC") is True  # Mixed case


def test_validate_commit_sha_invalid():
    """Test validation of invalid commit SHA."""
    # Test invalid SHAs
    assert dlml.validate_commit_sha("123456") is False  # Too short
    assert dlml.validate_commit_sha("12345678") is False  # Too long
    assert dlml.validate_commit_sha("123xyz!") is False  # Invalid characters


def test_validate_commit_sha_latest():
    """Test validation of the special 'latest' value."""
    assert dlml.validate_commit_sha("latest") is True


# --- File Hash Calculation Tests ---

def test_get_file_hash_existing_file():
    """Test hash calculation for existing files."""
    # Create a temporary file with known content
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(b"test content")
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


def test_get_file_hash_nonexistent_file():
    """Test hash calculation for non-existent files."""
    # Use a path that definitely doesn't exist
    nonexistent_path = "/path/that/definitely/does/not/exist"

    # Calculate hash
    file_hash = dlml.get_file_hash(nonexistent_path)

    # Verify hash is None
    assert file_hash is None


def test_get_file_hash_different_content():
    """Test hash calculation for files with different content."""
    # Create two temporary files with different content
    with tempfile.NamedTemporaryFile(delete=False) as temp_file1:
        temp_file1.write(b"content 1")
        temp_file1_path = temp_file1.name

    with tempfile.NamedTemporaryFile(delete=False) as temp_file2:
        temp_file2.write(b"content 2")
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


def test_get_file_hash_empty_file():
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
        assert file_hash == "d41d8cd98f00b204e9800998ecf8427e"
    finally:
        # Clean up
        os.unlink(temp_file_path)


# --- Cache Management Tests ---

def test_load_cache_valid_file():
    """Test loading from a valid cache file."""
    # Create a mock cache file
    cache_data = {"commit_sha": "1234567", "files": {"file1.txt": "hash1"}}
    mock_file = mock_open(read_data=json.dumps(cache_data))

    # Mock os.path.exists to return True
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_file):
        # Load cache
        loaded_cache = dlml.load_cache("cache_file.json")

        # Verify cache is loaded correctly
        assert loaded_cache == cache_data


def test_load_cache_nonexistent_file():
    """Test loading from a non-existent cache file."""
    # Mock os.path.exists to return False
    with patch("os.path.exists", return_value=False):
        # Load cache
        loaded_cache = dlml.load_cache("nonexistent_cache_file.json")

        # Verify empty cache is returned
        assert loaded_cache == {}


def test_load_cache_corrupted_file():
    """Test loading from a corrupted cache file."""
    # Create a mock corrupted cache file
    mock_file = mock_open(read_data="invalid json")

    # Mock os.path.exists to return True
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_file):
        # Load cache
        loaded_cache = dlml.load_cache("corrupted_cache_file.json")

        # Verify empty cache is returned
        assert loaded_cache == {}


def test_save_cache():
    """Test saving cache data to a file."""
    # Create mock cache data
    cache_data = {"commit_sha": "1234567", "files": {"file1.txt": "hash1"}}

    # Mock open and os.makedirs
    mock_file = mock_open()

    with patch("os.makedirs") as mock_makedirs, \
         patch("builtins.open", mock_file):
        # Save cache
        dlml.save_cache("cache_dir/cache_file.json", cache_data)

        # Verify os.makedirs was called with the correct arguments
        mock_makedirs.assert_called_once_with("cache_dir", exist_ok=True)

        # Verify open was called with the correct arguments
        mock_file.assert_called_once_with("cache_dir/cache_file.json", "w")

        # Verify json.dump was called with the correct arguments
        handle = mock_file()
        # json.dump writes in multiple chunks, so just verify write was called
        assert handle.write.call_count > 0


# --- File Download Tests ---

def test_download_file_success():
    """Test downloading a file from a valid URL."""
    # Create a mock response
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]

    # Mock requests.get to return the mock response
    with patch("requests.get", return_value=mock_response) as mock_get, \
         patch("os.makedirs") as mock_makedirs, \
         patch("builtins.open", mock_open()) as mock_file:
        # Download file
        dlml._download_file("http://example.com/file.txt", "dir/file.txt")

        # Verify requests.get was called with the correct arguments
        mock_get.assert_called_once_with("http://example.com/file.txt", stream=True)

        # Verify os.makedirs was called with the correct arguments
        mock_makedirs.assert_called_once_with("dir", exist_ok=True)

        # Verify open was called with the correct arguments
        mock_file.assert_called_once_with("dir/file.txt", "wb")

        # Verify write was called for each chunk
        handle = mock_file()
        assert handle.write.call_count == 2
        handle.write.assert_any_call(b"chunk1")
        handle.write.assert_any_call(b"chunk2")


def test_download_file_request_exception():
    """Test handling of download failures."""
    # Mock requests.get to raise an exception
    with patch("requests.get", side_effect=requests.exceptions.RequestException("Error")), \
         patch("os.makedirs"):
        # Download file and verify exception is raised
        with pytest.raises(RuntimeError) as excinfo:
            dlml._download_file("http://example.com/file.txt", "dir/file.txt")

        # Verify error message
        assert "Failed to download file: http://example.com/file.txt" in str(excinfo.value)


# --- DLML Download Function Tests ---

def test_download_data_files_with_version():
    """Test downloading with a specific version."""
    # Mock responses
    mock_release_response = MagicMock()
    mock_release_response.json.return_value = {"target_commitish": "1234567"}

    # Mock functions
    with patch("requests.get", return_value=mock_release_response) as mock_get, \
         patch("pelicun.tools.dlml._download_file") as mock_download, \
         patch("pelicun.tools.dlml.load_cache", return_value={}), \
         patch("pelicun.tools.dlml.save_cache"), \
         patch("pelicun.tools.dlml.get_file_hash", return_value="hash"), \
         patch("os.path.dirname", return_value="/path"), \
         patch("os.path.join", return_value="/path/to/file"), \
         patch("os.makedirs"), \
         patch("builtins.open", mock_open(read_data="file1.txt\nfile2.txt")), \
         patch("pelicun.tools.dlml.tqdm"):
        # Download data files
        dlml.download_data_files(version="v1.0.0")

        # Verify requests.get was called with the correct arguments
        mock_get.assert_called_once()
        assert "releases/tags/v1.0.0" in mock_get.call_args[0][0]

        # Verify _download_file was called for model_files.txt and each file in it
        assert mock_download.call_count == 3  # model_files.txt + 2 files


def test_download_data_files_with_commit():
    """Test downloading with a specific commit."""
    # Mock functions
    with patch("pelicun.tools.dlml.validate_commit_sha", return_value=True), \
         patch("pelicun.tools.dlml._download_file") as mock_download, \
         patch("pelicun.tools.dlml.load_cache", return_value={}), \
         patch("pelicun.tools.dlml.save_cache"), \
         patch("pelicun.tools.dlml.get_file_hash", return_value="hash"), \
         patch("os.path.dirname", return_value="/path"), \
         patch("os.path.join", return_value="/path/to/file"), \
         patch("os.makedirs"), \
         patch("builtins.open", mock_open(read_data="file1.txt\nfile2.txt")), \
         patch("pelicun.tools.dlml.tqdm"):
        # Download data files
        dlml.download_data_files(commit="1234567")

        # Verify _download_file was called for model_files.txt and each file in it
        assert mock_download.call_count == 3  # model_files.txt + 2 files


def test_download_data_files_with_latest_commit():
    """Test downloading with the 'latest' commit."""
    # Mock responses
    mock_commits_response = MagicMock()
    mock_commits_response.json.return_value = [{"sha": "1234567"}]

    # Mock functions
    with patch("requests.get", return_value=mock_commits_response) as mock_get, \
         patch("pelicun.tools.dlml._download_file") as mock_download, \
         patch("pelicun.tools.dlml.load_cache", return_value={}), \
         patch("pelicun.tools.dlml.save_cache"), \
         patch("pelicun.tools.dlml.get_file_hash", return_value="hash"), \
         patch("os.path.dirname", return_value="/path"), \
         patch("os.path.join", return_value="/path/to/file"), \
         patch("os.makedirs"), \
         patch("builtins.open", mock_open(read_data="file1.txt\nfile2.txt")), \
         patch("pelicun.tools.dlml.tqdm"):
        # Download data files
        dlml.download_data_files(commit="latest")

        # Verify requests.get was called with the correct arguments
        mock_get.assert_called_once()
        assert "commits" in mock_get.call_args[0][0]

        # Verify _download_file was called for model_files.txt and each file in it
        assert mock_download.call_count == 3  # model_files.txt + 2 files


def test_download_data_files_with_cache():
    """Test downloading with caching enabled."""
    # Mock cache with existing files
    mock_cache = {
        "commit_sha": "1234567",
        "files": {
            "file1.txt": "hash1",
            "file2.txt": "hash2"
        }
    }

    # Mock functions
    with patch("pelicun.tools.dlml.validate_commit_sha", return_value=True), \
         patch("pelicun.tools.dlml._download_file") as mock_download, \
         patch("pelicun.tools.dlml.load_cache", return_value=mock_cache), \
         patch("pelicun.tools.dlml.save_cache"), \
         patch("pelicun.tools.dlml.get_file_hash", return_value="hash1"), \
         patch("os.path.dirname", return_value="/path"), \
         patch("os.path.join", return_value="/path/to/file"), \
         patch("os.makedirs"), \
         patch("builtins.open", mock_open(read_data="file1.txt\nfile2.txt")), \
         patch("pelicun.tools.dlml.tqdm"):
        # Download data files with the same commit as in cache
        dlml.download_data_files(commit="1234567")

        # Verify _download_file was not called since files are in cache
        assert mock_download.call_count == 0


def test_download_data_files_invalid_commit():
    """Test handling of invalid versions or commits."""
    # Mock functions
    with patch("pelicun.tools.dlml.validate_commit_sha", return_value=False):
        # Download data files with invalid commit and verify exception is raised
        with pytest.raises(ValueError) as excinfo:
            dlml.download_data_files(commit="invalid")

        # Verify error message
        assert "Invalid commit SHA format" in str(excinfo.value)


def test_download_data_files_network_error():
    """Test handling of network errors during API calls."""
    # Mock requests.get to raise an exception
    with patch("requests.get", side_effect=requests.exceptions.RequestException("Error")), \
         patch("os.path.dirname", return_value="/path"), \
         patch("os.path.join", return_value="/path/to/file"), \
         patch("os.makedirs"):
        # Download data files and verify exception is raised
        with pytest.raises(RuntimeError) as excinfo:
            dlml.download_data_files(version="v1.0.0")

        # Verify error message
        assert "Failed to fetch release info" in str(excinfo.value)


# --- Error Handling Tests ---

def test_download_data_files_empty_repository():
    """Test handling when GitHub API returns empty commits list."""
    # Mock responses - empty commits list
    mock_commits_response = MagicMock()
    mock_commits_response.json.return_value = []  # Empty commits list

    # Mock functions
    with patch("requests.get", return_value=mock_commits_response) as mock_get, \
         patch("os.path.dirname", return_value="/path"), \
         patch("os.path.join", return_value="/path/to/file"), \
         patch("os.makedirs"):
        # Download data files with latest commit and verify exception is raised
        with pytest.raises(RuntimeError) as excinfo:
            dlml.download_data_files(commit="latest")

        # Verify error message
        assert "No commits found in the repository" in str(excinfo.value)


def test_download_data_files_missing_commit_sha_in_release():
    """Test handling when release data doesn't contain target_commitish."""
    # Mock responses - release data without target_commitish
    mock_release_response = MagicMock()
    mock_release_response.json.return_value = {"tag_name": "v1.0.0"}  # Missing target_commitish

    # Mock functions
    with patch("requests.get", return_value=mock_release_response) as mock_get, \
         patch("os.path.dirname", return_value="/path"), \
         patch("os.path.join", return_value="/path/to/file"), \
         patch("os.makedirs"):
        # Download data files and verify exception is raised
        with pytest.raises(RuntimeError) as excinfo:
            dlml.download_data_files(version="v1.0.0")

        # Verify error message
        assert "Could not find commit SHA for release 'v1.0.0'" in str(excinfo.value)


def test_download_data_files_model_file_not_found():
    """Test handling when model_files.txt doesn't exist after download."""
    # Mock responses
    mock_release_response = MagicMock()
    mock_release_response.json.return_value = {"target_commitish": "1234567"}

    # Mock functions - simulate FileNotFoundError when opening model_files.txt
    with patch("requests.get", return_value=mock_release_response), \
         patch("pelicun.tools.dlml._download_file"), \
         patch("pelicun.tools.dlml.load_cache", return_value={}), \
         patch("os.path.dirname", return_value="/path"), \
         patch("os.path.join", return_value="/path/to/file"), \
         patch("os.makedirs"), \
         patch("builtins.open", side_effect=FileNotFoundError("File not found")):
        # Download data files and verify exception is raised
        with pytest.raises(RuntimeError) as excinfo:
            dlml.download_data_files(version="v1.0.0")

        # Verify error message
        assert "Model file list not found at" in str(excinfo.value)


def test_download_data_files_model_file_read_error():
    """Test handling of general file reading exceptions."""
    # Mock responses
    mock_release_response = MagicMock()
    mock_release_response.json.return_value = {"target_commitish": "1234567"}

    # Mock functions - simulate general Exception when opening model_files.txt
    with patch("requests.get", return_value=mock_release_response), \
         patch("pelicun.tools.dlml._download_file"), \
         patch("pelicun.tools.dlml.load_cache", return_value={}), \
         patch("os.path.dirname", return_value="/path"), \
         patch("os.path.join", return_value="/path/to/file"), \
         patch("os.makedirs"), \
         patch("builtins.open", side_effect=Exception("General read error")):
        # Download data files and verify exception is raised
        with pytest.raises(RuntimeError) as excinfo:
            dlml.download_data_files(version="v1.0.0")

        # Verify error message
        assert "Error reading model file list" in str(excinfo.value)


# --- Progress Reporting Tests ---

def test_progress_bar_initialization():
    """Test progress bar initialization with different file counts."""
    # Mock tqdm
    with patch("pelicun.tools.dlml.tqdm") as mock_tqdm, \
         patch("pelicun.tools.dlml._download_file"), \
         patch("pelicun.tools.dlml.load_cache", return_value={}), \
         patch("pelicun.tools.dlml.save_cache"), \
         patch("pelicun.tools.dlml.get_file_hash", return_value="hash"), \
         patch("os.path.dirname", return_value="/path"), \
         patch("os.path.join", return_value="/path/to/file"), \
         patch("os.makedirs"), \
         patch("requests.get") as mock_get, \
         patch("builtins.open", mock_open(read_data="file1.txt\nfile2.txt")):
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"target_commitish": "1234567"}
        mock_get.return_value = mock_response

        # Download data files
        dlml.download_data_files(version="v1.0.0", use_cache=False)

        # Verify tqdm was called with the correct arguments
        mock_tqdm.assert_called_once()
        args, kwargs = mock_tqdm.call_args
        assert kwargs["total"] == 2  # 2 files in model_files.txt
        assert kwargs["desc"] == "Downloading files"
        assert kwargs["unit"] == "file"


def test_progress_bar_updates():
    """Test progress bar updates during downloads."""
    # Mock tqdm as a context manager
    mock_progress = MagicMock()
    mock_tqdm = MagicMock()
    mock_tqdm.return_value.__enter__.return_value = mock_progress

    with patch("pelicun.tools.dlml.tqdm", mock_tqdm), \
         patch("pelicun.tools.dlml._download_file"), \
         patch("pelicun.tools.dlml.load_cache", return_value={}), \
         patch("pelicun.tools.dlml.save_cache"), \
         patch("pelicun.tools.dlml.get_file_hash", return_value="hash"), \
         patch("os.path.dirname", return_value="/path"), \
         patch("os.path.join", return_value="/path/to/file"), \
         patch("os.makedirs"), \
         patch("requests.get") as mock_get, \
         patch("builtins.open", mock_open(read_data="file1.txt\nfile2.txt")):
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"target_commitish": "1234567"}
        mock_get.return_value = mock_response

        # Download data files
        dlml.download_data_files(version="v1.0.0", use_cache=False)

        # Verify progress bar was updated for each file
        assert mock_progress.update.call_count == 2  # Called once for each file
        assert mock_progress.set_description.call_count >= 2  # Called at least once for each file


def test_progress_reporting_for_skipped_files():
    """Test progress reporting for skipped files."""
    # Mock cache with existing files but different commit SHA to avoid early return
    mock_cache = {
        "commit_sha": "abcdefg",  # Different commit SHA
        "files": {
            "file1.txt": "hash1",
            "file2.txt": "hash2"
        }
    }

    # Mock tqdm as a context manager
    mock_progress = MagicMock()
    mock_tqdm = MagicMock()
    mock_tqdm.return_value.__enter__.return_value = mock_progress

    # Mock get_file_hash to return matching hashes for files to simulate caching
    def mock_get_file_hash_side_effect(path):
        # Return matching hashes for both files so they get skipped
        if path.endswith("file1.txt"):
            return "hash1"  # Matches cache, will be skipped
        elif path.endswith("file2.txt"):
            return "hash2"  # Matches cache, will be skipped
        return None  # For non-existent files

    # Mock os.path.join to return predictable paths
    def mock_path_join(*args):
        return "/".join(args)

    with patch("pelicun.tools.dlml.tqdm", mock_tqdm), \
         patch("pelicun.tools.dlml._download_file") as mock_download, \
         patch("pelicun.tools.dlml.load_cache", return_value=mock_cache), \
         patch("pelicun.tools.dlml.save_cache"), \
         patch("pelicun.tools.dlml.get_file_hash", side_effect=mock_get_file_hash_side_effect), \
         patch("os.path.dirname", return_value="/path"), \
         patch("os.path.join", side_effect=mock_path_join), \
         patch("os.makedirs"), \
         patch("os.path.exists", return_value=True), \
         patch("requests.get") as mock_get, \
         patch("builtins.open", mock_open(read_data="file1.txt\nfile2.txt")):
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"target_commitish": "1234567"}
        mock_get.return_value = mock_response

        # Download data files
        dlml.download_data_files(commit="1234567", use_cache=True)

        # Verify progress bar was updated for each file, even though they were skipped
        assert mock_progress.update.call_count == 2  # Called once for each file
        assert mock_download.call_count == 1  # Only model_files.txt was downloaded


# --- Command-Line Interface Tests ---

# Integration tests that verify the actual CLI interface works
def test_cli_integration_invalid_action():
    """Integration test: CLI with invalid action (not 'update')."""
    # Run the CLI with invalid action
    result = subprocess.run([
        "pelicun", "dlml", "invalid"
    ], capture_output=True, text=True)

    # Verify exit code and output
    assert result.returncode == 2  # argparse returns 2 for invalid choice
    assert "invalid choice: 'invalid'" in result.stderr


def test_cli_integration_missing_arguments():
    """Integration test: CLI with insufficient arguments."""
    # Run the CLI with no arguments for dlml
    result = subprocess.run([
        "pelicun", "dlml"
    ], capture_output=True, text=True)

    # Verify exit code and output
    assert result.returncode == 2  # argparse returns 2 for missing required argument
    assert "required" in result.stderr


# Unit tests for the main() function


def test_main_version_download():
    """Test successful version-based download via dlml_update() function."""
    with patch("pelicun.tools.dlml.download_data_files") as mock_download:
        # Test dlml_update function with version argument
        dlml.dlml_update(version='v1.0.0', use_cache=True)

        # Verify successful execution
        mock_download.assert_called_once_with(version='v1.0.0', use_cache=True)


def test_main_commit_download():
    """Test successful commit-based download via dlml_update() function."""
    with patch("pelicun.tools.dlml.download_data_files") as mock_download:
        # Test dlml_update function with commit argument
        dlml.dlml_update(commit='1234567', use_cache=True)

        # Verify successful execution
        mock_download.assert_called_once_with(commit='1234567', use_cache=True)


def test_main_no_cache_flag():
    """Test no-cache functionality via dlml_update() function."""
    with patch("pelicun.tools.dlml.download_data_files") as mock_download:
        # Test dlml_update function with use_cache=False
        dlml.dlml_update(version='v1.0.0', use_cache=False)

        # Verify successful execution
        mock_download.assert_called_once_with(version='v1.0.0', use_cache=False)


def test_main_download_error_handling():
    """Test error handling when download_data_files raises exceptions via dlml_update() function."""
    with patch("pelicun.tools.dlml.download_data_files", side_effect=RuntimeError("Test error")):
        # Test dlml_update function with mocked exception
        with pytest.raises(RuntimeError, match="Data download failed: Test error"):
            dlml.dlml_update(version='v1.0.0', use_cache=True)


def test_main_commit_with_no_cache():
    """Test dlml_update() function with commit and no cache."""
    with patch("pelicun.tools.dlml.download_data_files") as mock_download:
        # Test dlml_update function with commit and use_cache=False
        dlml.dlml_update(commit='1234567', use_cache=False)

        # Verify successful execution
        mock_download.assert_called_once_with(commit='1234567', use_cache=False)


def test_main_default_latest_version():
    """Test dlml_update() function with default 'latest' version when no version specified."""
    with patch("pelicun.tools.dlml.download_data_files") as mock_download:
        # Test dlml_update function with default latest version
        dlml.dlml_update()

        # Verify successful execution
        mock_download.assert_called_once_with(version='latest', use_cache=True)

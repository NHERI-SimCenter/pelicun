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

"""These are integration tests on the dlml module of pelicun."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pytest
import requests

from pelicun import dlml

def test_end_to_end_workflow_with_mock():
    """Test end-to-end workflow with mock GitHub API."""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock responses
        mock_release_response = MagicMock()
        mock_release_response.json.return_value = {"target_commitish": "1234567"}

        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]

        # Mock functions
        with patch("requests.get", return_value=mock_release_response) as mock_get, \
             patch("pelicun.dlml._download_file") as mock_download, \
             patch("os.path.dirname", return_value=temp_dir), \
             patch("builtins.open", mock_open(read_data="file1.txt\nfile2.txt")), \
             patch("pelicun.dlml.tqdm"):
            # Download data files
            dlml.download_data_files(version="v1.0.0", use_cache=False)

            # Verify requests.get was called with the correct arguments
            mock_get.assert_called_once()
            assert "releases/tags/v1.0.0" in mock_get.call_args[0][0]

            # Verify _download_file was called for model_files.txt and each file in it
            assert mock_download.call_count == 3  # model_files.txt + 2 files


def test_with_mock_github_api_responses():
    """Test with mock GitHub API responses."""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock responses for different API calls
        def mock_get_side_effect(url, **kwargs):
            if "releases/tags" in url:
                mock_response = MagicMock()
                mock_response.json.return_value = {"target_commitish": "1234567"}
                return mock_response
            elif "commits" in url:
                mock_response = MagicMock()
                mock_response.json.return_value = [{"sha": "1234567"}]
                return mock_response
            else:
                mock_response = MagicMock()
                mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
                return mock_response

        # Mock functions
        with patch("requests.get", side_effect=mock_get_side_effect) as mock_get, \
             patch("pelicun.dlml._download_file") as mock_download, \
             patch("os.path.dirname", return_value=temp_dir), \
             patch("builtins.open", mock_open(read_data="file1.txt\nfile2.txt")), \
             patch("pelicun.dlml.tqdm"):
            # Test with version
            dlml.download_data_files(version="v1.0.0", use_cache=False)
            assert mock_download.call_count == 3  # model_files.txt + 2 files
            mock_download.reset_mock()

            # Test with commit
            dlml.download_data_files(commit="1234567", use_cache=False)
            assert mock_download.call_count == 3  # model_files.txt + 2 files
            mock_download.reset_mock()

            # Test with latest commit
            dlml.download_data_files(commit="latest", use_cache=False)
            assert mock_download.call_count == 3  # model_files.txt + 2 files


# --- Edge Case Tests ---

def test_with_empty_model_files():
    """Test with empty model_files.txt."""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock responses
        mock_release_response = MagicMock()
        mock_release_response.json.return_value = {"target_commitish": "1234567"}

        # Mock functions
        with patch("requests.get", return_value=mock_release_response) as mock_get, \
             patch("pelicun.dlml._download_file") as mock_download, \
             patch("os.path.dirname", return_value=temp_dir), \
             patch("builtins.open", mock_open(read_data="")), \
             patch("pelicun.dlml.tqdm"):
            # Download data files
            dlml.download_data_files(version="v1.0.0", use_cache=False)

            # Verify _download_file was called only for model_files.txt
            assert mock_download.call_count == 1


def test_with_large_number_of_files():
    """Test with very large number of files."""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock responses
        mock_release_response = MagicMock()
        mock_release_response.json.return_value = {"target_commitish": "1234567"}

        # Create a large list of files
        large_file_list = "\n".join([f"file{i}.txt" for i in range(1000)])

        # Mock functions
        with patch("requests.get", return_value=mock_release_response) as mock_get, \
             patch("pelicun.dlml._download_file") as mock_download, \
             patch("os.path.dirname", return_value=temp_dir), \
             patch("builtins.open", mock_open(read_data=large_file_list)), \
             patch("pelicun.dlml.tqdm"):
            # Download data files
            dlml.download_data_files(version="v1.0.0", use_cache=False)

            # Verify _download_file was called for model_files.txt and each file in it
            assert mock_download.call_count == 1001  # model_files.txt + 1000 files


def test_with_special_characters_in_paths():
    """Test with special characters in file paths."""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock responses
        mock_release_response = MagicMock()
        mock_release_response.json.return_value = {"target_commitish": "1234567"}

        # Create a list of files with special characters
        special_file_list = "path/with spaces/file.txt\npath/with#hash/file.txt\npath/with?question/file.txt"

        # Mock functions
        with patch("requests.get", return_value=mock_release_response) as mock_get, \
             patch("pelicun.dlml._download_file") as mock_download, \
             patch("os.path.dirname", return_value=temp_dir), \
             patch("builtins.open", mock_open(read_data=special_file_list)), \
             patch("pelicun.dlml.tqdm"):
            # Download data files
            dlml.download_data_files(version="v1.0.0", use_cache=False)

            # Verify _download_file was called for model_files.txt and each file in it
            assert mock_download.call_count == 4  # model_files.txt + 3 files

            # Verify that the URLs contain the expected file paths
            call_args_list = [call[0][0] for call in mock_download.call_args_list]
            assert any("path/with spaces/file.txt" in url for url in call_args_list)
            assert any("path/with#hash/file.txt" in url for url in call_args_list)
            assert any("path/with?question/file.txt" in url for url in call_args_list)


def test_with_commented_lines_in_model_files():
    """Test with commented lines in model_files.txt."""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock responses
        mock_release_response = MagicMock()
        mock_release_response.json.return_value = {"target_commitish": "1234567"}

        # Create a list of files with commented lines
        file_list = """# This is a comment
file1.txt
# This is another comment
file2.txt
  # This is a comment with leading whitespace
file3.txt"""

        # Mock functions
        with patch("requests.get", return_value=mock_release_response) as mock_get, \
             patch("pelicun.dlml._download_file") as mock_download, \
             patch("os.path.dirname", return_value=temp_dir), \
             patch("builtins.open", mock_open(read_data=file_list)), \
             patch("pelicun.dlml.tqdm"):
            # Download data files
            dlml.download_data_files(version="v1.0.0", use_cache=False)

            # Verify _download_file was called for model_files.txt and each non-commented file
            assert mock_download.call_count == 5  # model_files.txt + 4 files

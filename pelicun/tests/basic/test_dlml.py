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

import importlib
import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, mock_open, patch

import pytest
import requests

from pelicun.tools import dlml


@pytest.fixture
def mock_download_env() -> Generator[dict[str, Any], None, None]:
    """Mocks the common environment for a data download test."""
    with patch('pelicun.tools.dlml.load_cache', return_value={}) as mock_load, patch(
        'pelicun.tools.dlml.save_cache'
    ) as mock_save, patch(
        'pelicun.tools.dlml.get_file_hash', return_value='hash'
    ) as mock_hash, patch('pathlib.Path.mkdir') as mock_mkdir, patch(
        'pelicun.tools.dlml.tqdm'
    ) as mock_tqdm:
        yield {  # noqa: DOC402
            'load': mock_load,
            'save': mock_save,
            'hash': mock_hash,
            'mkdir': mock_mkdir,
            'tqdm': mock_tqdm,
        }


# --- Commit SHA Validation Tests ---


@pytest.mark.parametrize(
    ('commit_sha', 'expected'),
    [
        # Valid cases
        ('1234567', True),
        ('abcdef0', True),
        ('7890ABC', True),  # Mixed case
        ('latest', True),  # Special case
        # Invalid cases
        ('123456', False),  # Too short
        ('12345678', False),  # Too long
        ('123xyz!', False),  # Invalid characters
    ],
)
def test_validate_commit_sha(commit_sha: str, expected: bool) -> None:  # noqa: FBT001
    """Test validation of commit SHA with various inputs."""
    assert dlml.validate_commit_sha(commit_sha) is expected


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
        assert len(file_hash) == 64  # SHA256 hash is 64 characters
    finally:
        # Clean up
        Path(temp_file_path).unlink()


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
        Path(temp_file1_path).unlink()
        Path(temp_file2_path).unlink()


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
        assert len(file_hash) == 64  # SHA256 hash is 64 characters

        # Known SHA256 hash for empty file
        assert (
            file_hash
            == 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
        )
    finally:
        # Clean up
        Path(temp_file_path).unlink()


# --- Cache Management Tests ---


def test_load_cache_valid_file() -> None:
    """Test loading from a valid cache file."""
    # Create a mock cache file
    cache_data = {'commit_sha': '1234567', 'files': {'file1.txt': 'hash1'}}
    mock_file = mock_open(read_data=json.dumps(cache_data))

    # Mock pathlib.Path.exists to return True and Path.open to use our mock
    with patch('pathlib.Path.exists', return_value=True), patch(
        'pathlib.Path.open', mock_file
    ):
        # Load cache
        loaded_cache = dlml.load_cache('cache_file.json')

        # Verify cache is loaded correctly
        assert loaded_cache == cache_data


def test_load_cache_nonexistent_file() -> None:
    """Test loading from a non-existent cache file."""
    # Mock pathlib.Path.exists to return False
    with patch('pathlib.Path.exists', return_value=False):
        # Load cache
        loaded_cache = dlml.load_cache('nonexistent_cache_file.json')

        # Verify empty cache is returned
        assert loaded_cache == {}


def test_load_cache_corrupted_file() -> None:
    """Test loading from a corrupted cache file."""
    # Create a mock corrupted cache file
    mock_file = mock_open(read_data='invalid json')

    # Mock pathlib.Path.exists to return True
    with patch('pathlib.Path.exists', return_value=True), patch(
        'pathlib.Path.open', mock_file
    ):
        # Load cache
        loaded_cache = dlml.load_cache('corrupted_cache_file.json')

        # Verify empty cache is returned
        assert loaded_cache == {}


def test_save_cache() -> None:
    """Test saving cache data to a file."""
    # Create mock cache data
    cache_data = {'commit_sha': '1234567', 'files': {'file1.txt': 'hash1'}}

    # Mock open and pathlib.Path.mkdir
    mock_file = mock_open()

    with patch('pathlib.Path.mkdir') as mock_mkdir, patch(
        'pathlib.Path.open', mock_file
    ):
        # Save cache
        dlml.save_cache('cache_dir/cache_file.json', cache_data)

        # Verify Path.mkdir was called with the correct arguments
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify Path.open was called with the correct arguments
        mock_file.assert_called_once_with('w')

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
        'pathlib.Path.mkdir'
    ) as mock_mkdir, patch('pathlib.Path.open', mock_open()) as mock_file:
        # Download file
        dlml._download_file('http://example.com/file.txt', 'dir/file.txt')

        # Verify requests.get was called with the correct arguments
        mock_get.assert_called_once_with(
            'http://example.com/file.txt', stream=True, timeout=10
        )

        # Verify Path.mkdir was called with the correct arguments
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify Path.open was called with the correct arguments
        mock_file.assert_called_once_with('wb')

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
    ), patch('pathlib.Path.mkdir'):
        # Download file and verify exception is raised
        with pytest.raises(RuntimeError) as excinfo:
            dlml._download_file('http://example.com/file.txt', 'dir/file.txt')

        # Verify error message
        assert 'Failed to download file: http://example.com/file.txt' in str(
            excinfo.value
        )


# --- Test Class for _resolve_version_to_commit_sha ---


class TestResolveVersionToCommitSHA:
    """Test class for the _resolve_version_to_commit_sha helper function."""

    def test_resolve_specific_version_tag(self) -> None:
        """Test resolving a specific version tag (e.g., 'v1.2.0')."""
        headers = {'Accept': 'application/vnd.github.v3+json'}

        mock_release_response = MagicMock()
        mock_release_response.json.return_value = {
            'target_commitish': '1234567',
            'tag_name': 'v1.2.0',
        }

        mock_tag_response = MagicMock()
        mock_tag_response.json.return_value = {
            'object': {'sha': 'abc1234567890', 'type': 'commit'}
        }

        def mock_get_side_effect(url: str, **kwargs) -> MagicMock:  # noqa: ARG001, ANN003
            if 'releases/tags/v1.2.0' in url:
                return mock_release_response
            if 'git/refs/tags/v1.2.0' in url:
                return mock_tag_response
            return MagicMock()

        with patch('requests.get', side_effect=mock_get_side_effect):
            commit_sha, cache_meta = dlml._resolve_version_to_commit_sha(
                headers, version='v1.2.0'
            )

            assert commit_sha == 'abc1234567890'
            assert cache_meta['version'] == 'v1.2.0'
            assert cache_meta['download_type'] == 'version'

    def test_resolve_latest_version_tag(self) -> None:
        """Test resolving the 'latest' version tag."""
        headers = {'Accept': 'application/vnd.github.v3+json'}

        mock_release_response = MagicMock()
        mock_release_response.json.return_value = {
            'target_commitish': '7654321',
            'tag_name': 'v2.0.0',
        }

        mock_tag_response = MagicMock()
        mock_tag_response.json.return_value = {
            'object': {'sha': 'def7654321098', 'type': 'commit'}
        }

        def mock_get_side_effect(url: str, **kwargs) -> MagicMock:  # noqa: ARG001, ANN003
            if 'releases/latest' in url:
                return mock_release_response
            if 'git/refs/tags/v2.0.0' in url:
                return mock_tag_response
            return MagicMock()

        with patch('requests.get', side_effect=mock_get_side_effect):
            commit_sha, cache_meta = dlml._resolve_version_to_commit_sha(
                headers, version='latest'
            )

            assert commit_sha == 'def7654321098'
            assert cache_meta['version'] == 'v2.0.0'
            assert cache_meta['download_type'] == 'version'

    def test_resolve_annotated_tag(self) -> None:
        """Test handling annotated tags (key new test)."""
        headers = {'Accept': 'application/vnd.github.v3+json'}

        mock_release_response = MagicMock()
        mock_release_response.json.return_value = {
            'target_commitish': 'main',
            'tag_name': 'v1.5.0',
        }

        mock_tag_response = MagicMock()
        mock_tag_response.json.return_value = {
            'object': {'sha': 'annotated123', 'type': 'tag'}  # Annotated tag
        }

        mock_tag_object_response = MagicMock()
        mock_tag_object_response.json.return_value = {
            'object': {'sha': 'commit456789', 'type': 'commit'}
        }

        def mock_get_side_effect(url: str, **kwargs) -> MagicMock:  # noqa: ARG001, ANN003
            if 'releases/tags/v1.5.0' in url:
                return mock_release_response
            if 'git/refs/tags/v1.5.0' in url:
                return mock_tag_response
            if 'git/tags/annotated123' in url:
                return mock_tag_object_response
            return MagicMock()

        with patch('requests.get', side_effect=mock_get_side_effect):
            commit_sha, cache_meta = dlml._resolve_version_to_commit_sha(
                headers, version='v1.5.0'
            )

            assert commit_sha == 'commit456789'
            assert cache_meta['version'] == 'v1.5.0'
            assert cache_meta['download_type'] == 'version'

    def test_resolve_specific_commit_sha(self) -> None:
        """Test resolving a specific commit SHA."""
        headers = {'Accept': 'application/vnd.github.v3+json'}

        with patch('pelicun.tools.dlml.validate_commit_sha', return_value=True):
            commit_sha, cache_meta = dlml._resolve_version_to_commit_sha(
                headers, commit='abc1234'
            )

            assert commit_sha == 'abc1234'
            assert cache_meta['version'] == 'commit-abc1234'
            assert cache_meta['download_type'] == 'commit'

    def test_resolve_latest_commit(self) -> None:
        """Test resolving the 'latest' commit."""
        headers = {'Accept': 'application/vnd.github.v3+json'}

        mock_commits_response = MagicMock()
        mock_commits_response.json.return_value = [
            {'sha': 'latest123456789'},
            {'sha': 'older987654321'},
        ]

        with patch('requests.get', return_value=mock_commits_response):
            commit_sha, cache_meta = dlml._resolve_version_to_commit_sha(
                headers, commit='latest'
            )

            assert commit_sha == 'latest123456789'
            assert cache_meta['version'] == 'commit-latest1'
            assert cache_meta['download_type'] == 'commit'

    def test_invalid_commit_sha_format_raises_value_error(self) -> None:
        """Test gracefully handling ValueError for invalid commit SHA formats."""
        headers = {'Accept': 'application/vnd.github.v3+json'}

        with (
            patch('pelicun.tools.dlml.validate_commit_sha', return_value=False),
            pytest.raises(ValueError, match='Invalid commit SHA format'),
        ):
            dlml._resolve_version_to_commit_sha(headers, commit='invalid')

    def test_api_call_failure_raises_runtime_error(self) -> None:
        """Test gracefully handling RuntimeError when API calls fail."""
        headers = {'Accept': 'application/vnd.github.v3+json'}

        # Test network error for version resolution
        with (
            patch(
                'requests.get',
                side_effect=requests.exceptions.RequestException('Network error'),
            ),
            pytest.raises(RuntimeError, match='Failed to fetch release info'),
        ):
            dlml._resolve_version_to_commit_sha(headers, version='v1.0.0')

        # Test network error for latest commit resolution
        with (
            patch(
                'requests.get',
                side_effect=requests.exceptions.RequestException('Network error'),
            ),
            pytest.raises(RuntimeError, match='Failed to fetch latest commit'),
        ):
            dlml._resolve_version_to_commit_sha(headers, commit='latest')

    def test_empty_commits_list_raises_runtime_error(self) -> None:
        """Test handling empty commits list when fetching latest commit."""
        headers = {'Accept': 'application/vnd.github.v3+json'}

        mock_commits_response = MagicMock()
        mock_commits_response.json.return_value = []  # Empty commits list

        with (
            patch('requests.get', return_value=mock_commits_response),
            pytest.raises(RuntimeError, match='No commits found in the repository'),
        ):
            dlml._resolve_version_to_commit_sha(headers, commit='latest')

    def test_missing_commit_sha_in_release_raises_runtime_error(self) -> None:
        """Test handling missing commit SHA in release data."""
        headers = {'Accept': 'application/vnd.github.v3+json'}

        mock_release_response = MagicMock()
        mock_release_response.json.return_value = {
            'tag_name': 'v1.0.0',
            # Missing target_commitish
        }

        mock_tag_response = MagicMock()
        mock_tag_response.side_effect = requests.exceptions.RequestException(
            'Tag API failed'
        )

        def mock_get_side_effect(url: str, **kwargs) -> MagicMock:  # noqa: ARG001, ANN003
            if 'releases/tags/v1.0.0' in url:
                return mock_release_response
            if 'git/refs/tags/v1.0.0' in url:
                raise requests.exceptions.RequestException('Tag API failed')  # noqa: EM101
            return MagicMock()

        with (
            patch('requests.get', side_effect=mock_get_side_effect),
            pytest.raises(
                RuntimeError, match="Could not find commit SHA for release 'v1.0.0'"
            ),
        ):
            dlml._resolve_version_to_commit_sha(headers, version='v1.0.0')

    def test_fallback_to_target_commitish_on_tag_api_failure(self) -> None:
        """Test fallback to target_commitish when tag API fails."""
        headers = {'Accept': 'application/vnd.github.v3+json'}

        mock_release_response = MagicMock()
        mock_release_response.json.return_value = {
            'target_commitish': 'fallback123456',
            'tag_name': 'v1.0.0',
        }

        def mock_get_side_effect(url: str, **kwargs) -> MagicMock:  # noqa: ARG001, ANN003
            if 'releases/tags/v1.0.0' in url:
                return mock_release_response
            if 'git/refs/tags/v1.0.0' in url:
                raise requests.exceptions.RequestException('Tag API failed')  # noqa: EM101
            return MagicMock()

        with patch('requests.get', side_effect=mock_get_side_effect):
            commit_sha, cache_meta = dlml._resolve_version_to_commit_sha(
                headers, version='v1.0.0'
            )

            assert commit_sha == 'fallback123456'
            assert cache_meta['version'] == 'v1.0.0'
            assert cache_meta['download_type'] == 'version'


# --- DLML Download Function Tests ---


def test_download_data_files_with_version(mock_download_env: dict[str, Any]) -> None:  # noqa: ARG001
    """Test downloading with a specific version."""
    with patch(
        'pelicun.tools.dlml._resolve_version_to_commit_sha',
        return_value=(
            'abc1234567890',
            {'version': 'v1.0.0', 'download_type': 'version'},
        ),
    ), patch('pelicun.tools.dlml._download_file') as mock_download, patch(
        'pathlib.Path.open', mock_open(read_data='file1.txt\nfile2.txt')
    ):
        dlml.download_data_files(version='v1.0.0')
        assert mock_download.call_count == 3


def test_download_data_files_with_commit(mock_download_env: dict[str, Any]) -> None:  # noqa: ARG001
    """Test downloading with a specific commit."""
    with patch(
        'pelicun.tools.dlml._resolve_version_to_commit_sha',
        return_value=(
            '1234567',
            {'version': 'commit-1234567', 'download_type': 'commit'},
        ),
    ), patch('pelicun.tools.dlml._download_file') as mock_download, patch(
        'pathlib.Path.open', mock_open(read_data='file1.txt\nfile2.txt')
    ):
        dlml.download_data_files(commit='1234567')
        assert mock_download.call_count == 3


def test_download_data_files_with_latest_commit(
    mock_download_env: dict[str, Any],  # noqa: ARG001
) -> None:
    """Test downloading with the 'latest' commit."""
    with patch(
        'pelicun.tools.dlml._resolve_version_to_commit_sha',
        return_value=(
            'latest123456789',
            {'version': 'commit-latest1', 'download_type': 'commit'},
        ),
    ), patch('pelicun.tools.dlml._download_file') as mock_download, patch(
        'pathlib.Path.open', mock_open(read_data='file1.txt\nfile2.txt')
    ):
        dlml.download_data_files(commit='latest')
        assert mock_download.call_count == 3


def test_download_data_files_with_cache(mock_download_env: dict[str, Any]) -> None:
    """Test downloading with caching enabled."""
    mock_cache = {
        'commit_sha': '1234567',
        'files': {'file1.txt': 'hash1', 'file2.txt': 'hash2'},
    }

    # Override the fixture's load_cache to return our mock_cache
    mock_download_env['load'].return_value = mock_cache
    # Override the fixture's get_file_hash to return matching hash
    mock_download_env['hash'].return_value = 'hash1'

    with patch(
        'pelicun.tools.dlml._resolve_version_to_commit_sha',
        return_value=(
            '1234567',
            {'version': 'commit-1234567', 'download_type': 'commit'},
        ),
    ), patch('pelicun.tools.dlml._download_file') as mock_download, patch(
        'pathlib.Path.open', mock_open(read_data='file1.txt\nfile2.txt')
    ):
        dlml.download_data_files(commit='1234567')

        assert mock_download.call_count == 0


# --- Error Handling Tests ---


def test_download_data_files_model_file_not_found() -> None:
    """Test handling when model_files.txt doesn't exist after download."""
    # Mock functions - simulate FileNotFoundError when opening model_files.txt
    with patch(
        'pelicun.tools.dlml._resolve_version_to_commit_sha',
        return_value=('1234567', {'version': 'v1.0.0', 'download_type': 'version'}),
    ), patch('pelicun.tools.dlml._download_file'), patch(
        'pelicun.tools.dlml.load_cache', return_value={}
    ), patch('pathlib.Path.mkdir'), patch(
        'pathlib.Path.open', side_effect=FileNotFoundError('File not found')
    ):
        # Download data files and verify exception is raised
        with pytest.raises(RuntimeError) as excinfo:
            dlml.download_data_files(version='v1.0.0')

        # Verify error message
        assert 'Model file list not found at' in str(excinfo.value)


def test_download_data_files_model_file_read_error() -> None:
    """Test handling of general file reading exceptions."""
    # Mock functions - simulate general Exception when opening model_files.txt
    with patch(
        'pelicun.tools.dlml._resolve_version_to_commit_sha',
        return_value=('1234567', {'version': 'v1.0.0', 'download_type': 'version'}),
    ), patch('pelicun.tools.dlml._download_file'), patch(
        'pelicun.tools.dlml.load_cache', return_value={}
    ), patch('pathlib.Path.mkdir'), patch(
        'pathlib.Path.open', side_effect=Exception('General read error')
    ):
        # Download data files and verify exception is raised
        with pytest.raises(RuntimeError) as excinfo:
            dlml.download_data_files(version='v1.0.0')

        # Verify error message
        assert 'Error reading model file list' in str(excinfo.value)


# --- Progress Reporting Tests ---


def test_progress_bar_initialization(mock_download_env: dict[str, Any]) -> None:
    """Test progress bar initialization with different file counts."""
    with patch(
        'pelicun.tools.dlml._resolve_version_to_commit_sha',
        return_value=('1234567', {'version': 'v1.0.0', 'download_type': 'version'}),
    ), patch('pelicun.tools.dlml._download_file'), patch(
        'pathlib.Path.open', mock_open(read_data='file1.txt\nfile2.txt')
    ):
        dlml.download_data_files(version='v1.0.0', use_cache=False)

        # Verify tqdm was called with the correct arguments
        mock_download_env['tqdm'].assert_called_once()
        _args, kwargs = mock_download_env['tqdm'].call_args
        assert kwargs['total'] == 2  # 2 files in model_files.txt
        assert kwargs['desc'] == 'Downloading files'
        assert kwargs['unit'] == 'file'


def test_progress_bar_updates(mock_download_env: dict[str, Any]) -> None:
    """Test progress bar updates during downloads."""
    # Mock tqdm as a context manager
    mock_progress = MagicMock()
    mock_download_env['tqdm'].return_value.__enter__.return_value = mock_progress

    with patch(
        'pelicun.tools.dlml._resolve_version_to_commit_sha',
        return_value=('1234567', {'version': 'v1.0.0', 'download_type': 'version'}),
    ), patch('pelicun.tools.dlml._download_file'), patch(
        'pathlib.Path.open', mock_open(read_data='file1.txt\nfile2.txt')
    ):
        dlml.download_data_files(version='v1.0.0', use_cache=False)

        # Verify progress bar was updated for each file
        assert mock_progress.update.call_count == 2  # Called once for each file
        assert (
            mock_progress.set_description.call_count >= 2
        )  # Called at least once for each file


def test_progress_reporting_for_skipped_files(
    mock_download_env: dict[str, Any],
) -> None:
    """Test progress reporting for skipped files."""
    # Mock cache with existing files but different commit SHA to avoid early return
    mock_cache = {
        'commit_sha': 'abcdefg',  # Different commit SHA
        'files': {'file1.txt': 'hash1', 'file2.txt': 'hash2'},
    }

    # Override the fixture's load_cache to return our mock_cache
    mock_download_env['load'].return_value = mock_cache

    # Mock tqdm as a context manager
    mock_progress = MagicMock()
    mock_download_env['tqdm'].return_value.__enter__.return_value = mock_progress

    # Mock get_file_hash to return matching hashes for files to simulate caching
    def mock_get_file_hash_side_effect(path: str) -> str:
        # Return matching hashes for both files so they get skipped
        if path.endswith('file1.txt'):
            return 'hash1'  # Matches cache, will be skipped
        if path.endswith('file2.txt'):
            return 'hash2'  # Matches cache, will be skipped
        return None  # For non-existent files

    # Override the fixture's get_file_hash with our side effect
    mock_download_env['hash'].side_effect = mock_get_file_hash_side_effect

    with (
        patch(
            'pelicun.tools.dlml._resolve_version_to_commit_sha',
            return_value=(
                '1234567',
                {'version': 'commit-1234567', 'download_type': 'commit'},
            ),
        ),
        patch('pelicun.tools.dlml._get_changed_files', return_value=set()),
        patch('pelicun.tools.dlml._download_file') as mock_download,
        patch('pathlib.Path.open', mock_open(read_data='file1.txt\nfile2.txt')),
    ):
        # Download data files
        dlml.download_data_files(commit='1234567', use_cache=True)

        # Verify progress bar was updated for each file, even though they were skipped
        assert mock_progress.update.call_count == 2  # Called once for each file
        assert mock_download.call_count == 1  # model_files.txt


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
    ), patch('pathlib.Path.exists', return_value=True):
        result = dlml.check_dlml_version()

        # Should return cached result without making API call
        assert result['update_available'] is True
        assert result['current_version'] == 'v1.0.0'
        assert result['latest_version'] == 'v1.1.0'
        assert result['last_check'] == recent_time.isoformat()
        assert result['error'] is None


def test_check_dlml_version_with_expired_cache() -> None:
    """Test check_dlml_version performs new check when cache is expired."""

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
        'pathlib.Path.exists', return_value=True
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
    ), patch('pathlib.Path.exists', return_value=True):
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
        'pathlib.Path.exists', return_value=True
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
        'pathlib.Path.exists', return_value=True
    ):
        result = dlml.check_dlml_version()

        # Should always suggest update for commit-based versions
        assert result['update_available'] is True
        assert result['current_version'] == 'commit-abc1234'
        assert result['latest_version'] == 'v1.1.0'


def test_check_dlml_data_cli_invocation_bypass() -> None:
    """Test check_dlml_data returns immediately when invoked via CLI for dlml update."""
    # Test various CLI command patterns that should trigger bypass
    cli_patterns = [
        ['pelicun', 'dlml', 'update'],
        ['python', '-m', 'pelicun', 'dlml', 'update'],
        ['/path/to/pelicun', 'dlml', 'update', '--version', 'latest'],
        ['cli.py', 'dlml', 'update'],
    ]

    for argv_pattern in cli_patterns:
        with patch('sys.argv', argv_pattern), patch(
            'pathlib.Path.exists'
        ) as mock_exists, patch(
            'pelicun.tools.dlml.download_data_files'
        ) as mock_download:
            dlml.check_dlml_data()

            # Should return immediately without checking paths or downloading
            mock_exists.assert_not_called()
            mock_download.assert_not_called()


def test_check_dlml_data_with_empty_directory() -> None:
    """
    Test check_dlml_data with an empty temporary directory.

    This mocks DLML_DATA_DIR to use a temporary directory, ensuring the
    function triggers a download when data is missing. It also reloads the
    module to ensure the original, un-mocked function is tested.
    """
    # Reload the module to bypass potential mocks from other tests.
    importlib.reload(dlml)

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)

        with patch('pelicun.tools.dlml.DLML_DATA_DIR', temp_path), patch(
            'pelicun.tools.dlml.download_data_files'
        ) as mock_download, patch('builtins.print'):
            dlml.check_dlml_data()

            mock_download.assert_called_once_with(version='latest', use_cache=False)


def test_check_dlml_data_with_existing_data_no_update() -> None:
    """Test check_dlml_data with existing data and no update available."""
    # Mock version check result - no update available
    mock_version_info = {
        'update_available': False,
        'current_version': 'v1.0.0',
        'latest_version': 'v1.0.0',
        'error': None,
    }

    with patch('pathlib.Path.exists', return_value=True), patch(
        'pathlib.Path.is_dir', return_value=True
    ), patch(
        'pathlib.Path.iterdir',
        return_value=[MagicMock(spec=Path, name='model1.json')],
    ), patch(
        'pelicun.tools.dlml.check_dlml_version', return_value=mock_version_info
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

    with patch('pathlib.Path.exists', return_value=True), patch(
        'pathlib.Path.is_dir', return_value=True
    ), patch(
        'pathlib.Path.iterdir',
        return_value=[MagicMock(spec=Path, name='model1.json')],
    ), patch(
        'pelicun.tools.dlml.check_dlml_version', return_value=mock_version_info
    ), patch('warnings.warn') as mock_warn:
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
    with patch('pathlib.Path.exists', return_value=False), patch(
        'pelicun.tools.dlml.download_data_files',
        side_effect=requests.exceptions.ConnectionError('Network error'),
    ):
        # Should raise RuntimeError with detailed message
        with pytest.raises(RuntimeError) as exc_info:
            dlml.check_dlml_data()

        error_msg = str(exc_info.value)
        assert 'Network error while downloading DLML data' in error_msg
        assert 'pelicun dlml update' in error_msg


def test_check_dlml_data_permission_error() -> None:
    """Test check_dlml_data handles permission errors appropriately."""
    with patch('pathlib.Path.exists', return_value=False), patch(
        'pelicun.tools.dlml.download_data_files',
        side_effect=PermissionError('Access denied'),
    ):
        # Should raise RuntimeError with permission-specific message
        with pytest.raises(RuntimeError) as exc_info:
            dlml.check_dlml_data()

        error_msg = str(exc_info.value)
        assert 'Permission error while downloading DLML data' in error_msg
        assert 'permissions' in error_msg


def test_check_dlml_data_version_check_failure() -> None:
    """Test check_dlml_data handles version check failures gracefully."""
    # Reload the module to bypass potential mocks from other tests.
    importlib.reload(dlml)

    with patch('sys.argv', ['test_script.py']), patch(
        'pathlib.Path.exists', return_value=True
    ), patch('pathlib.Path.is_dir', return_value=True), patch(
        'pathlib.Path.iterdir',
        return_value=[MagicMock(spec=Path, name='model1.json')],
    ), patch(
        'pelicun.tools.dlml.check_dlml_version',
        side_effect=ValueError('Version check failed'),
    ), patch('builtins.print') as mock_print:
        # Should not raise exception, just print the error.
        dlml.check_dlml_data()

        # It should call print() to report the failure.
        mock_print.assert_called_once()
        print_msg = mock_print.call_args[0][0]
        assert 'Version check failed' in print_msg


# --- Tests for _format_initial_download_error ---


@pytest.mark.parametrize(
    ('exception', 'expected_messages'),
    [
        (
            requests.exceptions.ConnectionError('Connection failed'),
            [
                'Network error while downloading DLML data',
                'Connection failed',
                'Please check your internet connection and try again.',
            ],
        ),
        (
            PermissionError('Access denied'),
            [
                'Permission error while downloading DLML data',
                'Access denied',
                'Please check file/directory permissions for the pelicun installation.',
            ],
        ),
        (
            ValueError('Invalid value provided'),
            ['Error downloading DLML data (ValueError)', 'Invalid value provided'],
        ),
    ],
)
def test_format_initial_download_error(
    exception: Exception, expected_messages: list[str]
) -> None:
    """Test _format_initial_download_error with various exception types."""
    result = dlml._format_initial_download_error(exception)

    for expected_msg in expected_messages:
        assert expected_msg in result


def test_check_dlml_version_invalid_version_format() -> None:
    """Test check_dlml_version handles invalid version format in API response."""
    mock_cache = {'version': 'v1.0.0'}

    # Mock GitHub API response with invalid semantic version
    mock_response = MagicMock()
    mock_response.json.return_value = {'tag_name': 'v2.0-beta'}
    mock_response.raise_for_status.return_value = None

    with patch('pelicun.tools.dlml.load_cache', return_value=mock_cache), patch(
        'pelicun.tools.dlml.save_cache'
    ), patch('requests.get', return_value=mock_response), patch(
        'pathlib.Path.exists', return_value=True
    ):
        result = dlml.check_dlml_version()

        # Should fall back to string comparison when version parsing fails
        assert (
            result['update_available'] is True
        )  # Different strings, so update available
        assert result['current_version'] == 'v1.0.0'
        assert result['latest_version'] == 'v2.0-beta'
        assert result['error'] is None


def test_check_dlml_version_corrupted_timestamp() -> None:
    """Test check_dlml_version handles corrupted timestamp in cache."""
    # Mock cache with corrupted timestamp
    mock_cache = {
        'version': 'v1.0.0',
        'last_version_check': 'invalid-timestamp-format',  # This will cause ValueError/TypeError
    }

    # Mock GitHub API response
    mock_response = MagicMock()
    mock_response.json.return_value = {'tag_name': 'v1.1.0'}
    mock_response.raise_for_status.return_value = None

    with patch('pelicun.tools.dlml.load_cache', return_value=mock_cache), patch(
        'pelicun.tools.dlml.save_cache'
    ), patch('requests.get', return_value=mock_response), patch(
        'pathlib.Path.exists', return_value=True
    ):
        result = dlml.check_dlml_version()

        # Should ignore corrupted timestamp and make fresh API call
        assert result['update_available'] is True
        assert result['current_version'] == 'v1.0.0'
        assert result['latest_version'] == 'v1.1.0'
        assert result['error'] is None


def test_download_data_files_empty_model_files() -> None:
    """Test download_data_files handles empty model_files.txt gracefully."""
    # Mock empty model_files.txt content (only comments and whitespace)
    empty_model_files_content = """# This is a comment
# Another comment


# Final comment
"""

    with patch(
        'pelicun.tools.dlml._resolve_version_to_commit_sha'
    ) as mock_resolve, patch(
        'pelicun.tools.dlml._download_file'
    ) as mock_download_file, patch(
        'pelicun.tools.dlml.load_cache', return_value={}
    ), patch('pelicun.tools.dlml.save_cache'), patch(
        'pelicun.tools.dlml.get_file_hash', return_value='hash123'
    ), patch(
        'pathlib.Path.open', mock_open(read_data=empty_model_files_content)
    ), patch('pathlib.Path.mkdir'):
        # Mock version resolution
        mock_resolve.return_value = (
            'abc1234',
            {'version': 'v1.0.0', 'download_type': 'release'},
        )

        # Call the function
        dlml.download_data_files(version='v1.0.0', use_cache=True)

        # Verify model_files.txt was downloaded
        assert mock_download_file.call_count == 1
        download_call = mock_download_file.call_args_list[0]
        assert (
            'model_files.txt' in download_call[0][0]
        )  # URL contains model_files.txt

        # Verify no additional files were downloaded (only model_files.txt)
        # The function should complete gracefully without downloading any data files


def test_download_data_files_corrupted_local_file_smart_update() -> None:
    """Test download_data_files re-downloads corrupted local file during smart update."""
    # Mock model_files.txt content with one file
    model_files_content = 'file1.txt\n'

    # Mock cache data representing v2.0 installation
    old_cache = {
        'commit_sha': 'old1234',
        'files': {
            'file1.txt': 'old_hash_123'  # Hash in cache
        },
    }

    with patch(
        'pelicun.tools.dlml._resolve_version_to_commit_sha'
    ) as mock_resolve, patch(
        'pelicun.tools.dlml._download_file'
    ) as mock_download_file, patch(
        'pelicun.tools.dlml._get_changed_files'
    ) as mock_get_changed, patch(
        'pelicun.tools.dlml.load_cache', return_value=old_cache
    ), patch('pelicun.tools.dlml.save_cache'), patch(
        'pelicun.tools.dlml.get_file_hash'
    ) as mock_get_hash, patch(
        'pathlib.Path.open', mock_open(read_data=model_files_content)
    ), patch('pathlib.Path.mkdir'):
        # Mock version resolution (updating from v2.0 to v2.1)
        mock_resolve.return_value = (
            'new5678',
            {'version': 'v2.1', 'download_type': 'release'},
        )

        # Mock _get_changed_files to report file1.txt as NOT changed
        mock_get_changed.return_value = set()  # Empty set means no files changed

        # Mock get_file_hash to return different hash (corrupted local file)
        mock_get_hash.return_value = (
            'corrupted_hash_456'  # Different from cached hash
        )

        # Call the function
        dlml.download_data_files(version='v2.1', use_cache=True)

        # Verify model_files.txt was downloaded
        assert mock_download_file.call_count == 2  # model_files.txt + file1.txt

        # Verify file1.txt was re-downloaded despite being "unchanged"
        download_calls = [call[0][0] for call in mock_download_file.call_args_list]
        assert any('model_files.txt' in url for url in download_calls)
        assert any('file1.txt' in url for url in download_calls)

        # Verify the hash check was performed
        mock_get_hash.assert_called()


class TestGetChangedFiles:
    """A dedicated test class for the _get_changed_files function."""

    def test_get_changed_files_upgrade(self) -> None:
        """Test the standard upgrade path where commits are ahead."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'status': 'ahead',
            'files': [{'filename': 'file1.txt'}, {'filename': 'file2.json'}],
        }

        with patch('requests.get', return_value=mock_response) as mock_get:
            changed_files = dlml._get_changed_files(
                'base_commit', 'head_commit', headers={}
            )

            # Ensure the correct API URL was called
            mock_get.assert_called_once()
            call_url = mock_get.call_args[0][0]
            assert 'base_commit...head_commit' in call_url

            # Verify the returned filenames are correct
            assert changed_files == {'file1.txt', 'file2.json'}

    def test_get_changed_files_downgrade(self) -> None:
        """Test the downgrade path where commits are behind, triggering a second API call."""
        # Mock response for the initial, 'behind' call
        mock_behind_response = MagicMock()
        mock_behind_response.json.return_value = {'status': 'behind', 'files': []}

        # Mock response for the second, reversed call that contains the file list
        mock_ahead_response = MagicMock()
        mock_ahead_response.json.return_value = {
            'status': 'ahead',
            'files': [{'filename': 'file_to_revert.csv'}],
        }

        # The side_effect will return the 'behind' response first, then the 'ahead' one
        with patch(
            'requests.get', side_effect=[mock_behind_response, mock_ahead_response]
        ) as mock_get:
            changed_files = dlml._get_changed_files(
                'new_commit', 'old_commit', headers={}
            )

            # Check that the API was called twice
            assert mock_get.call_count == 2

            # Check the first call was in the original order
            first_call_url = mock_get.call_args_list[0][0][0]
            assert 'new_commit...old_commit' in first_call_url

            # Check the second call was in the reversed order
            second_call_url = mock_get.call_args_list[1][0][0]
            assert 'old_commit...new_commit' in second_call_url

            # The final result should be from the second call
            assert changed_files == {'file_to_revert.csv'}

    def test_get_changed_files_api_failure(self) -> None:
        """Test the function's behavior when the GitHub API call fails."""
        with patch(
            'requests.get',
            side_effect=requests.exceptions.RequestException('Network Error'),
        ), patch.object(dlml.logger, 'warning') as mock_warning:
            changed_files = dlml._get_changed_files(
                'base_commit', 'head_commit', headers={}
            )

            # On failure, the function should return None and log a warning
            assert changed_files is None
            mock_warning.assert_called()

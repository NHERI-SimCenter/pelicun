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


"""Methods for handling the Damage and Loss Model Library."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shlex
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests
from packaging import version
from tqdm import tqdm

from pelicun.pelicun_warnings import PelicunWarning

# Configure logger once at the module level
logger = logging.getLogger('pelicun.dlml')

# --- Module-level path constants ---
PELICUN_ROOT_DIR = Path(__file__).resolve().parent.parent


def get_api_repo_url() -> str:
    """
    Construct the base URL for the GitHub API repository endpoint.

    Returns
    -------
    str
        The base URL for the GitHub API repository endpoint.
    """
    owner = os.environ.get('DLML_REPO_OWNER', 'NHERI-SimCenter')
    repo = os.environ.get('DLML_REPO_NAME', 'DamageAndLossModelLibrary')
    api_base = os.environ.get('PELICUN_GITHUB_API_URL', 'https://api.github.com')
    return f'{api_base}/repos/{owner}/{repo}'


def get_raw_repo_url() -> str:
    """
    Construct the base URL for the GitHub raw content endpoint.

    Returns
    -------
    str
        The base URL for the GitHub raw content endpoint.
    """
    owner = os.environ.get('DLML_REPO_OWNER', 'NHERI-SimCenter')
    repo = os.environ.get('DLML_REPO_NAME', 'DamageAndLossModelLibrary')
    raw_base = os.environ.get(
        'PELICUN_GITHUB_RAW_URL', 'https://raw.githubusercontent.com'
    )
    return f'{raw_base}/{owner}/{repo}'


def get_dlml_data_dir() -> Path:
    """
    Get the DLML data directory path.

    Checks for the DLML_DATA_DIR environment variable for overrides,
    otherwise defaults to the standard location within the package.

    Returns
    -------
    Path
        The DLML data directory path.
    """
    if data_dir_override := os.environ.get('DLML_DATA_DIR'):
        return Path(data_dir_override)
    return PELICUN_ROOT_DIR / 'resources' / 'DamageAndLossModelLibrary'


def validate_commit_sha(commit: str) -> bool:
    """
    Validate that the commit SHA is in the correct format.

    Parameters
    ----------
    commit: string
        The commit SHA to validate

    Returns
    -------
    bool
        True if the commit is valid, False otherwise
    """
    if commit == 'latest':
        return True

    # GitHub short SHA is typically 7 characters
    pattern = r'^[0-9a-f]{7}$'
    return bool(re.match(pattern, commit, re.IGNORECASE))


def get_file_hash(file_path: Union[str, Path]) -> Optional[str]:  # noqa: UP007
    """
    Calculate the SHA256 hash of a file.

    Parameters
    ----------
    file_path: string
        Path to the file

    Returns
    -------
    string
        SHA256 hash of the file, or None if the file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        return None

    with path.open('rb') as f:
        file_hash = hashlib.sha256()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def _get_changed_files(
    base_commit: str, head_commit: str, headers: dict
) -> set[str] | None:
    """
    Get a set of filenames that have changed between two commits.

    This function intelligently handles both upgrades and downgrades by
    checking the commit status and reversing the comparison if necessary.

    Parameters
    ----------
    base_commit: str
        The starting commit SHA.
    head_commit: str
        The ending commit SHA.
    headers: dict
        Headers to use for the GitHub API request.

    Returns
    -------
    set[str] | None
        A set of changed filenames, or None if the API call fails.
    """
    compare_url = f'{get_api_repo_url()}/compare/{base_commit}...{head_commit}'

    try:
        response = requests.get(compare_url, headers=headers, timeout=(10, 60))
        response.raise_for_status()

        compare_data = response.json()

        # Check the status. 'behind' means we are downgrading.
        if compare_data.get('status') == 'behind':
            # If we're downgrading, the diff is empty in this direction.
            # We must swap the commits to find the files that need to be reverted.
            reverse_compare_url = (
                f'{get_api_repo_url()}/compare/{head_commit}...{base_commit}'
            )
            response = requests.get(
                reverse_compare_url, headers=headers, timeout=(10, 60)
            )
            response.raise_for_status()
            compare_data = response.json()

        return {file_info['filename'] for file_info in compare_data.get('files', [])}

    except requests.exceptions.RequestException as e:
        logger.warning(f'Could not get commit comparison from GitHub API: {e}')
        logger.warning('Will proceed by re-downloading all necessary files.')
        return None


def load_cache(cache_file: Union[str, Path]) -> Dict[str, Any]:  # noqa: UP007, UP006
    """
    Load the cache from a file.

    Parameters
    ----------
    cache_file: string
        Path to the cache file

    Returns
    -------
    dict
        The cache data, or an empty dict if the file doesn't exist
    """
    path = Path(cache_file)
    if path.exists():
        try:
            with path.open('r') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            # If the cache file is corrupted, return an empty cache
            return {}
    return {}


def save_cache(cache_file: Union[str, Path], cache_data: Dict[str, Any]) -> None:  # noqa: UP007, UP006
    """
    Save the cache to a file.

    Parameters
    ----------
    cache_file: string
        Path to the cache file
    cache_data: dict
        The cache data to save
    """
    path = Path(cache_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(cache_data, f)


def check_dlml_version() -> Dict[str, Union[bool, str, None]]:  # noqa: UP007, UP006, C901
    """
    Check if there's a newer version of DLML data available.

    This function checks daily for new releases and returns version information.

    Returns
    -------
    dict
        Dictionary containing:
        - 'update_available': bool, True if update is available
        - 'current_version': str, current local version or None
        - 'latest_version': str, latest available version or None
        - 'last_check': str, ISO timestamp of last check
        - 'error': str, error message if check failed
    """
    target_data_abs_dir = get_dlml_data_dir()

    # Cache file path
    cache_file = target_data_abs_dir / '.dlml_cache.json'

    # Load existing cache
    cache_data = load_cache(str(cache_file))

    # Check if we need to perform a version check (daily)
    now = datetime.now()  # noqa: DTZ005
    last_version_check = cache_data.get('last_version_check')

    if last_version_check:
        try:
            last_check_time = datetime.fromisoformat(last_version_check)
            if now - last_check_time < timedelta(days=1):
                # Return cached version check result
                return {
                    'update_available': cache_data.get('update_available', False),
                    'current_version': cache_data.get('current_version'),
                    'latest_version': cache_data.get('latest_version'),
                    'last_check': last_version_check,
                    'error': cache_data.get('version_check_error'),
                }
        except (ValueError, TypeError):
            # Invalid timestamp, proceed with check
            pass

    # Perform version check
    result = {
        'update_available': False,
        'current_version': None,
        'latest_version': None,
        'last_check': now.isoformat(),
        'error': None,
    }

    try:
        # Get current local version from cache
        current_version_str = cache_data.get('version')
        if current_version_str:
            # Handle commit-based versions by extracting the last version before that commit
            if current_version_str.startswith('commit-'):
                # For commit-based installs, we can't easily compare versions
                # Use the commit SHA as the current version identifier
                result['current_version'] = current_version_str
            else:
                result['current_version'] = current_version_str

        # Get latest version from GitHub releases
        headers = {'Accept': 'application/vnd.github.v3+json'}

        # Check for the GITHUB_TOKEN environment variable
        token = os.environ.get('GITHUB_TOKEN')
        if token:
            headers['Authorization'] = f'Bearer {token}'

        release_url = f'{get_api_repo_url()}/releases/latest'
        response = requests.get(release_url, headers=headers, timeout=(10, 60))
        response.raise_for_status()

        release_data = response.json()
        latest_version_str = release_data.get('tag_name')

        if latest_version_str:
            result['latest_version'] = latest_version_str

            # Compare versions if we have both
            if current_version_str and not current_version_str.startswith('commit-'):
                try:
                    current_ver = version.parse(current_version_str.lstrip('v'))
                    latest_ver = version.parse(latest_version_str.lstrip('v'))
                    result['update_available'] = latest_ver > current_ver
                except version.InvalidVersion:
                    # If version parsing fails, assume update is available
                    result['update_available'] = (
                        current_version_str != latest_version_str
                    )
            elif current_version_str and current_version_str.startswith('commit-'):
                # For commit-based versions, always suggest checking for updates
                result['update_available'] = True
            elif not current_version_str:
                # No local version info, suggest update
                result['update_available'] = True

    except requests.exceptions.RequestException as e:
        result['error'] = f'Failed to check for updates: {e}'
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as e:
        result['error'] = f'Version check error: {e}'

    # Update cache with version check results
    cache_data.update(
        {
            'last_version_check': result['last_check'],
            'update_available': result['update_available'],
            'current_version': result['current_version'],
            'latest_version': result['latest_version'],
            'version_check_error': result['error'],
        }
    )

    # Save updated cache
    if target_data_abs_dir.exists():
        save_cache(str(cache_file), cache_data)

    return result


def _resolve_version_to_commit_sha(  # noqa: C901
    headers: dict,
    version: str | None = None,
    commit: str | None = None,
) -> tuple[str, dict]:
    """
    Resolve a version tag or commit SHA to a full commit SHA.

    Returns a tuple containing the commit SHA and metadata for caching.

    Parameters
    ----------
    headers: dict
        A dictionary of headers to be used in the request.
    version: string, optional
        A version tag (e.g., "v1.0.0", "beta"). Use "latest" to download from
        the latest published release.
    commit: string, optional
        A commit SHA (7-digit identifier). Use "latest" to download from the
        latest commit.

    Returns
    -------
    tuple
        commit_sha: string
            The full commit SHA for the resolved version.
        commit_meta: dict
            A dictionary containing metadata for caching.

    Raises
    ------
    ValueError
        If an invalid commit SHA format is provided
    RuntimeError
        If data download fails due to network issues, missing commits, or file errors
    """
    # Logic for handling 'latest' commit or a specific commit SHA
    if commit is not None:
        if commit != 'latest' and not validate_commit_sha(commit):
            msg = (
                f'Invalid commit SHA format: {commit}. '
                f"Must be 'latest' or a 7-character hexadecimal string."
            )
            raise ValueError(msg)

        if commit == 'latest':
            logger.info('Downloading DLML models from the latest commit...')

            commits_url = f'{get_api_repo_url()}/commits'
            try:
                response = requests.get(
                    commits_url, headers=headers, timeout=(10, 60)
                )
                response.raise_for_status()
                commits_data = response.json()
                if commits_data and len(commits_data) > 0:
                    commit_sha = commits_data[0]['sha']
                else:
                    msg = 'No commits found in the repository.'
                    raise RuntimeError(msg)
            except requests.exceptions.RequestException as e:
                msg = f'Failed to fetch latest commit: {e}'
                raise RuntimeError(msg) from e
        else:
            logger.info(f'Downloading DLML models from commit {commit}...')
            commit_sha = commit

        cache_meta = {
            'version': f'commit-{commit_sha[:7]}',
            'download_type': 'commit',
        }
        return commit_sha, cache_meta

    # Logic for handling version tags
    version_str = version or 'latest'
    release_info_url = (
        f'{get_api_repo_url()}/releases/tags/{version_str}'
        if version_str != 'latest'
        else f'{get_api_repo_url()}/releases/latest'
    )

    try:
        response = requests.get(release_info_url, headers=headers, timeout=(10, 60))
        response.raise_for_status()
        release_data = response.json()
    except requests.exceptions.RequestException as e:
        msg = f"Failed to fetch release info for '{version_str}': {e}"
        raise RuntimeError(msg) from e

    # Get the SHA of the commit the release tag points to
    # Note: target_commitish can be a branch name (like "main"), so we need to
    # get the actual commit SHA that the tag points to using the Git refs API
    tag_name = release_data.get('tag_name', version_str)
    tag_refs_url = f'{get_api_repo_url()}/git/refs/tags/{tag_name}'

    try:
        # Primary method: resolve the tag via Git refs API
        tag_response = requests.get(tag_refs_url, headers=headers, timeout=(10, 60))
        tag_response.raise_for_status()
        tag_data = tag_response.json()
        tag_sha = tag_data.get('object', {}).get('sha')
        tag_type = tag_data.get('object', {}).get('type')

        if tag_type == 'commit':  # Tag points directly to a commit
            commit_sha = tag_sha
        elif tag_type == 'tag':  # Annotated tag
            tag_object_url = f'{get_api_repo_url()}/git/tags/{tag_sha}'
            tag_obj_response = requests.get(
                tag_object_url, headers=headers, timeout=(10, 60)
            )
            tag_obj_response.raise_for_status()
            commit_sha = tag_obj_response.json().get('object', {}).get('sha')
        else:  # Fallback
            commit_sha = release_data.get('target_commitish')
    except requests.exceptions.RequestException:
        # Fallback if API fails
        commit_sha = release_data.get('target_commitish')

    if not commit_sha:
        msg = f"Could not find commit SHA for release '{version_str}'."
        raise RuntimeError(msg)

    cache_meta = {
        'version': (
            version
            if version != 'latest'
            else release_data.get('tag_name', 'latest')
        ),
        'download_type': 'version',
    }
    return commit_sha, cache_meta


def _download_file(url: str, local_path: Union[str, Path]) -> None:  # noqa: UP007
    """
    Download a file from a GitHub repository using a direct URL.

    Parameters
    ----------
    url: string
        URL pointing to the file to download
    local_path: string
        Local path where the file should be downloaded

    Raises
    ------
    RuntimeError
        If the file download fails due to network or other issues
    """
    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, stream=True, timeout=(10, 60))
        response.raise_for_status()

        with path.open('wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    except requests.exceptions.RequestException as e:
        msg = f'Failed to download file: {url}'
        raise RuntimeError(msg) from e


def download_data_files(  # noqa: C901
    version: str = 'latest',
    commit: Optional[str] = None,  # noqa: UP007
    use_cache: bool = True,  # noqa: FBT001, FBT002
) -> None:
    """
    Download model files from DLML based on the model_files.txt.

    Parameters
    ----------
    version: string
        The release tag name (e.g., "v1.0.0", "beta"). Use "latest" to download
        from the latest published release. Ignored if commit is provided.
    commit: string, optional
        The commit SHA (7-digit identifier). Use "latest" to download from the latest commit.
        If provided, this takes precedence over the version parameter.
    use_cache: bool, optional
        Whether to use caching to avoid re-downloading unchanged files. Default is True.

    Raises
    ------
    RuntimeError
        If data download fails due to network issues, missing commits, or file errors
    """
    target_data_abs_dir = get_dlml_data_dir()
    target_data_abs_dir.mkdir(parents=True, exist_ok=True)

    # Cache file path
    cache_file = target_data_abs_dir / '.dlml_cache.json'

    # Load cache if using caching
    cache_data = load_cache(str(cache_file)) if use_cache else {}

    headers = {'Accept': 'application/vnd.github.v3+json'}
    # Check for the GITHUB_TOKEN environment variable
    token = os.environ.get('GITHUB_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'

    try:
        commit_sha, cache_meta = _resolve_version_to_commit_sha(
            headers, version=version, commit=commit
        )
    except (requests.exceptions.RequestException, RuntimeError, ValueError) as e:
        msg = f'Failed to determine download version: {e}'
        raise RuntimeError(msg) from e

    changed_files = None
    if use_cache:
        old_commit_sha = cache_data.get('commit_sha')

        # Check if we already have the latest data
        if old_commit_sha == commit_sha:
            logger.info(f'Already have the latest data for commit {commit_sha[:7]}.')
            return

        # Determine which files have changed if we are updating
        if old_commit_sha:
            logger.info(
                f'Updating from commit {old_commit_sha[:7]} to {commit_sha[:7]}.'
            )
            changed_files = _get_changed_files(old_commit_sha, commit_sha, headers)

    # Get the model file list
    model_file_list_url = f'{get_raw_repo_url()}/{commit_sha}/model_files.txt'

    model_file_local_path = target_data_abs_dir / 'model_files.txt'
    _download_file(model_file_list_url, str(model_file_local_path))

    logger.info('Successfully downloaded model file list.')

    # Read the model list and download model files
    try:
        with model_file_local_path.open('r') as f:
            files_to_download = [
                line.strip()
                for line in f
                if line.strip() and not line.startswith('#')
            ]
    except FileNotFoundError as e:
        msg = f'Model file list not found at {model_file_local_path}'
        raise RuntimeError(msg) from e
    except Exception as e:
        msg = f'Error reading model file list: {e}'
        raise RuntimeError(msg) from e

    if changed_files is not None:
        relevant_changed_files = {
            file for file in files_to_download if file in changed_files
        }
        logger.info(f'Found {len(relevant_changed_files)} relevant file changes.')

    file_download_base_url = f'{get_raw_repo_url()}/{commit_sha}'

    # Initialize cache for this commit if using caching
    if use_cache:
        # Keep track of files that should exist
        new_cache = {
            'commit_sha': commit_sha,
            'last_updated': datetime.now().isoformat(),  # noqa: DTZ005
            'files': {},
        }

        # Add version metadata for tracking
        new_cache['version'] = cache_meta['version']
        new_cache['download_type'] = cache_meta['download_type']

    # Count total files for progress reporting
    total_files = len(files_to_download)
    logger.info(f'Checking {total_files} files for updates.')

    # Download files with progress reporting using tqdm
    skipped_count = 0

    # Create a progress bar for all files
    with tqdm(total=total_files, desc='Downloading files', unit='file') as pbar:
        for file_path in files_to_download:
            # Update progress bar description to show current file
            local_path = target_data_abs_dir / file_path
            pbar.set_description(f'Processing {local_path.name}')

            # Check if we can skip this file
            # A file is considered "in cache" and can be skipped if:
            # 1. The API call for changed files succeeded.
            # 2. The file is NOT in the list of changed files.
            # 3. The file is correctly represented in our old cache.
            # 4. The local file hash matches the old cache hash (integrity check).
            is_unchanged_on_remote = (changed_files is not None) and (
                file_path not in changed_files
            )
            if use_cache and is_unchanged_on_remote:
                cached_hash = cache_data.get('files', {}).get(file_path)
                if cached_hash:
                    current_hash = get_file_hash(str(local_path))
                    if (
                        current_hash == cached_hash
                    ):  # File is valid in cache, skip download
                        skipped_count += 1
                        new_cache['files'][file_path] = current_hash
                        pbar.update(1)
                        continue  # Move to the next file

            # Download the file if it's not in cache
            pbar.set_description(f'Downloading {local_path.name}')
            remote_url = f'{file_download_base_url}/{file_path}'
            _download_file(remote_url, str(local_path))

            if use_cache:
                new_cache['files'][file_path] = get_file_hash(str(local_path))

            pbar.update(1)

    # Save the new cache
    if use_cache:
        save_cache(str(cache_file), new_cache)

    logger.info(
        f'DLML model data download complete. Downloaded {total_files - skipped_count} files, skipped {skipped_count} unchanged files.'
    )


def _format_initial_download_error(e: Exception) -> str:
    """
    Format a detailed error message for a failed initial download.

    Parameters
    ----------
    e: Exception
        The exception that was raised during the initial download.

    Returns
    -------
    error_msg: string
        A formatted error message for the initial download failure.

    """
    error_type = type(e).__name__
    error_str = str(e).lower()

    if 'requests' in str(type(e)).lower() or 'connection' in error_str:
        return (
            f'Network error while downloading DLML data: {e}\n'
            'Please check your internet connection and try again.'
        )
    if 'permission' in error_str or 'access' in error_str:
        return (
            f'Permission error while downloading DLML data: {e}\n'
            'Please check file/directory permissions for the pelicun installation.'
        )

    return f'Error downloading DLML data ({error_type}): {e}'


def check_dlml_data() -> None:
    """
    Ensure DLML data is available for pelicun, downloading if necessary.

    This function is called during pelicun import to check for DLML data availability.
    If data is not found, it downloads the latest version. If data exists, it performs
    a daily version check and warns if updates are available.

    Raises
    ------
    RuntimeError
        If the initial data download fails since pelicun cannot function without DLML data.
    """
    # If pelicun is imported to perform a DLML update, return immediately and
    # let the CLI command handle all operations.
    search_space = sys.argv[:11]
    if ('dlml', 'update') in zip(search_space, search_space[1:]):
        return

    target_data_abs_dir = get_dlml_data_dir()

    # Check if the critical manifest file exists as a proxy for a valid installation
    manifest_file = target_data_abs_dir / 'model_files.txt'
    data_exists = manifest_file.exists()

    # Construct a robust update command
    safe_python_exe = shlex.quote(sys.executable)
    update_command = f'{safe_python_exe} -m pelicun.cli dlml update'

    if not data_exists:
        # No DLML data found, download latest version
        try:
            print('DLML model data not found. Downloading latest version...')  # noqa: T201
            print('This is a one-time setup that may take a few minutes.')  # noqa: T201
            download_data_files(version='latest', use_cache=True)
            print('DLML data download completed successfully.')  # noqa: T201
        except Exception as e:
            detailed_msg = _format_initial_download_error(e)
            final_msg = (
                f'{detailed_msg}\n'
                'Pelicun requires DLML data to function properly.\n'
                f'You can manually download using: {update_command}'
            )
            raise RuntimeError(final_msg) from e
    else:
        # Data exists, perform version check
        try:
            version_info = check_dlml_version()

            if version_info.get('update_available') and not version_info.get(
                'error'
            ):
                current_ver = version_info.get('current_version', 'unknown')
                latest_ver = version_info.get('latest_version', 'unknown')

                warning_msg = (
                    f'DLML data update available. '
                    f'Current version: {current_ver}, '
                    f'Latest version: {latest_ver}. '
                    f'Update with: {update_command}'
                )

                # Use pelicun's warning system
                warnings.warn(warning_msg, PelicunWarning, stacklevel=2)

        except (
            requests.exceptions.RequestException,
            ValueError,
            TypeError,
            KeyError,
            json.JSONDecodeError,
        ) as e:
            # Version check failed, but don't prevent import
            print(f'Version check failed: {e}')  # noqa: T201


def dlml_update(
    version: Optional[str] = None,  # noqa: UP007
    commit: Optional[str] = None,  # noqa: UP007
    use_cache: bool = True,  # noqa: FBT001, FBT002
) -> None:
    """
    Update DLML data files.

    Parameters
    ----------
    version: str, optional
        Version tag (e.g., 'v1.2.0') or 'latest'. Default is 'latest'.
    commit: str, optional
        Commit SHA to download from. If provided, version is ignored.
    use_cache: bool, optional
        Whether to use caching to avoid re-downloading unchanged files. Default is True.

    Raises
    ------
    RuntimeError
        If the data download fails.
    """
    try:
        if commit is not None:
            # Handle commit-based download
            download_data_files(commit=commit, use_cache=use_cache)
        else:
            # Handle version-based download
            version_arg = version if version is not None else 'latest'
            download_data_files(version=version_arg, use_cache=use_cache)
    except (RuntimeError, ValueError) as e:
        msg = f'Data download failed: {e}'
        raise RuntimeError(msg) from e

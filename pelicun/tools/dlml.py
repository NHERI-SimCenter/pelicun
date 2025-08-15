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
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests
from packaging import version
from tqdm import tqdm

# Configure logger once at module level
logger = logging.getLogger('pelicun.dlml')

# --- Configuration for your data repository ---
DATA_REPO_OWNER = 'zsarnoczay'

# --- Module-level path constants ---
PELICUN_ROOT_DIR = Path(__file__).resolve().parent.parent
DLML_DATA_DIR = PELICUN_ROOT_DIR / 'resources' / 'DamageAndLossModelLibrary'


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
    Calculate the MD5 hash of a file.

    Parameters
    ----------
    file_path: string
        Path to the file

    Returns
    -------
    string
        MD5 hash of the file, or None if the file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        return None

    with path.open('rb') as f:
        file_hash = hashlib.md5()
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def load_cache(cache_file: Union[str, Path]) -> Dict[str, Any]:
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
        except (json.JSONDecodeError, IOError):
            # If the cache file is corrupted, return an empty cache
            return {}
    return {}


def save_cache(cache_file: Union[str, Path], cache_data: Dict[str, Any]) -> None:
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


def check_dlml_version() -> Dict[str, Union[bool, str, None]]:  # noqa: UP007
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
    target_data_abs_dir = DLML_DATA_DIR

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
        GitHub_api_base_url = f'https://api.github.com/repos/{DATA_REPO_OWNER}/DamageAndLossModelLibrary'
        headers = {'Accept': 'application/vnd.github.v3+json'}

        release_url = f'{GitHub_api_base_url}/releases/latest'
        response = requests.get(release_url, headers=headers, timeout=10)
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
    except Exception as e:
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


def _download_file(url: str, local_path: Union[str, Path]) -> None:
    """
    Download a file from a GitHub repository using a direct URL.

    Parameters
    ----------
    url: string
        URL pointing to the file to download
    local_path: string
        Local path where the file should be downloaded
    """
    path = Path(local_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with path.open('wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    except requests.exceptions.RequestException as e:
        msg = f'Failed to download file: {url}'
        raise RuntimeError(msg) from e


def download_data_files(
    version: str = 'latest',
    commit: Optional[str] = None,
    use_cache: bool = True,  # noqa: UP007
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
    """
    target_data_abs_dir = DLML_DATA_DIR
    target_data_abs_dir.mkdir(parents=True, exist_ok=True)

    # Cache file path
    cache_file = target_data_abs_dir / '.dlml_cache.json'

    # Load cache if using caching
    cache_data = load_cache(str(cache_file)) if use_cache else {}

    # Determine if we're using a commit or a version
    commit_sha = None
    GitHub_api_base_url = (
        f'https://api.github.com/repos/{DATA_REPO_OWNER}/DamageAndLossModelLibrary'
    )
    headers = {'Accept': 'application/vnd.github.v3+json'}

    if commit is not None:
        # Validate commit SHA format
        if commit != 'latest' and not validate_commit_sha(commit):
            msg = (
                f'Invalid commit SHA format: {commit}. '
                f"Must be 'latest' or a 7-character hexadecimal string."
            )
            raise ValueError(msg)

        if commit == 'latest':
            # Get the latest commit SHA
            logger.info(
                f'Downloading DLML models from the latest commit to {target_data_abs_dir}...'
            )
            commits_url = f'{GitHub_api_base_url}/commits'
            try:
                response = requests.get(commits_url, headers=headers)
                response.raise_for_status()
                commits_data = response.json()
                if commits_data and len(commits_data) > 0:
                    commit_sha = commits_data[0]['sha']
                else:
                    raise RuntimeError('No commits found in the repository.')
            except requests.exceptions.RequestException as e:
                msg = f'Failed to fetch latest commit: {e}'
                raise RuntimeError(msg) from e
        else:
            # Use the provided commit SHA
            logger.info(
                f'Downloading DLML models from commit {commit} to {target_data_abs_dir}...'
            )
            commit_sha = commit
    else:
        # Use version-based download (original behavior)
        logger.info(
            f'Downloading DLML models for version {version} to {target_data_abs_dir}...'
        )
        release_info_url = (
            f'{GitHub_api_base_url}/releases/tags/{version}'
            if version != 'latest'
            else f'{GitHub_api_base_url}/releases/latest'
        )

        try:
            response = requests.get(release_info_url, headers=headers)
            response.raise_for_status()
            release_data = response.json()
        except requests.exceptions.RequestException as e:
            msg = f"Failed to fetch release info for '{version}': {e}"
            raise RuntimeError(msg) from e

        # Get the SHA of the commit the release tag points to
        commit_sha = release_data.get('target_commitish')
        if not commit_sha:
            msg = f"Could not find commit SHA for release '{version}'."
            raise RuntimeError(msg)

    # Check if we already have the latest data
    if (
        use_cache
        and 'commit_sha' in cache_data
        and cache_data['commit_sha'] == commit_sha
    ):
        logger.info(f'Already have the latest data for commit {commit_sha[:7]}.')
        return

    # Get the model file list
    model_file_list_url = f'https://raw.githubusercontent.com/{DATA_REPO_OWNER}/DamageAndLossModelLibrary/{commit_sha}/model_files.txt'

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
    except FileNotFoundError:
        msg = f'Model file list not found at {model_file_local_path}'
        raise RuntimeError(msg)
    except Exception as e:
        msg = f'Error reading model file list: {e}'
        raise RuntimeError(msg)

    file_download_base_url = f'https://raw.githubusercontent.com/{DATA_REPO_OWNER}/DamageAndLossModelLibrary/{commit_sha}'

    # Initialize cache for this commit if using caching
    if use_cache:
        # Keep track of files that should exist
        new_cache = {
            'commit_sha': commit_sha,
            'last_updated': datetime.now().isoformat(),  # noqa: DTZ005
            'files': {},
        }

        # Add version metadata for tracking
        if commit is not None:
            # For commit-based downloads, store commit info
            new_cache['version'] = f'commit-{commit_sha[:7]}'
            new_cache['download_type'] = 'commit'
        else:
            # For version-based downloads, store the version tag
            new_cache['version'] = (
                version
                if version != 'latest'
                else release_data.get('tag_name', 'latest')
            )
            new_cache['download_type'] = 'version'

    # Count total files for progress reporting
    total_files = len(files_to_download)
    logger.info(f'Found {total_files} files to download.')

    # Download files with progress reporting using tqdm
    skipped_count = 0

    # Create a progress bar for all files
    with tqdm(total=total_files, desc='Downloading files', unit='file') as pbar:
        for file_path in files_to_download:
            remote_url = f'{file_download_base_url}/{file_path}'
            local_path = target_data_abs_dir / file_path

            # Update progress bar description to show current file
            pbar.set_description(f'Downloading {local_path.name}')

            # Check if file exists in cache and hasn't changed
            file_in_cache = False
            if (
                use_cache
                and 'files' in cache_data
                and file_path in cache_data['files']
            ):
                # Get the current hash if the file exists
                current_hash = get_file_hash(str(local_path))
                cached_hash = cache_data['files'][file_path]

                if current_hash and current_hash == cached_hash:
                    # File exists and hasn't changed, skip download
                    file_in_cache = True
                    skipped_count += 1

                    # Add to new cache
                    if use_cache:
                        new_cache['files'][file_path] = current_hash

                    # Update progress bar
                    pbar.update(1)

            if not file_in_cache:
                # Download the file
                _download_file(remote_url, str(local_path))

                # Add to new cache
                if use_cache:
                    new_cache['files'][file_path] = get_file_hash(str(local_path))

                # Update progress bar
                pbar.update(1)

    # Save the new cache
    if use_cache:
        save_cache(str(cache_file), new_cache)

    logger.info(
        f'DLML model data download complete. Downloaded {total_files - skipped_count} files, skipped {skipped_count} unchanged files.'
    )


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
    import warnings

    from pelicun.pelicun_warnings import PelicunWarning

    target_data_abs_dir = DLML_DATA_DIR

    # Check if DLML data directory exists and has content
    data_exists = (
        target_data_abs_dir.exists()
        and target_data_abs_dir.is_dir()
        and len(list(target_data_abs_dir.iterdir())) > 0
    )

    if not data_exists:
        # No DLML data found, download latest version
        try:
            logger.info('DLML model data not found. Downloading latest version...')
            logger.info('This is a one-time setup that may take a few minutes.')
            download_data_files(version='latest', use_cache=True)
            logger.info('DLML data download completed successfully.')
        except Exception as e:
            # Provide detailed, error-specific messages
            error_type = type(e).__name__
            if 'requests' in str(type(e)).lower() or 'connection' in str(e).lower():
                detailed_msg = (
                    f'Network error while downloading DLML data: {e}\n'
                    f'Please check your internet connection and try again.\n'
                    f'If the problem persists, you can manually download using: pelicun dlml update'
                )
            elif 'permission' in str(e).lower() or 'access' in str(e).lower():
                detailed_msg = (
                    f'Permission error while downloading DLML data: {e}\n'
                    f'Please check file/directory permissions for the pelicun installation.\n'
                    f'You may need to run with appropriate permissions or manually download using: pelicun dlml update'
                )
            else:
                detailed_msg = (
                    f'Error downloading DLML data ({error_type}): {e}\n'
                    f'Pelicun requires DLML data to function properly.\n'
                    f'You can manually download using: pelicun dlml update'
                )
            raise RuntimeError(detailed_msg) from e
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
                    f'Update with: pelicun dlml update'
                )

                # Use pelicun's warning system
                warnings.warn(warning_msg, PelicunWarning, stacklevel=2)

        except Exception as e:
            # Version check failed, but don't prevent import
            logger.debug(f'Version check failed: {e}')


def dlml_update(
    version: Optional[str] = None,  # noqa: UP007
    commit: Optional[str] = None,  # noqa: UP007
    use_cache: bool = True,
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
    ValueError
        If invalid parameters are provided.
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

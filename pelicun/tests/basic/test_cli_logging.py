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

"""These are unit tests for the CLI logging functionality of pelicun."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pelicun.cli import setup_dlml_logging


def test_setup_dlml_logging_stdout_only() -> None:
    """Test setup_dlml_logging with stdout logging only."""
    # Clear any existing handlers
    logger = logging.getLogger('pelicun.dlml')
    logger.handlers.clear()

    # Setup logging without file
    setup_dlml_logging(log_file=None)

    # Verify logger configuration
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1

    # Verify handler is StreamHandler
    handler = logger.handlers[0]
    assert isinstance(handler, logging.StreamHandler)

    # Verify formatter
    formatter = handler.formatter
    assert formatter._fmt == '%(message)s'

    # Clean up
    logger.handlers.clear()


def test_setup_dlml_logging_with_file() -> None:
    """Test setup_dlml_logging with file logging."""
    # Clear any existing handlers
    logger = logging.getLogger('pelicun.dlml')
    logger.handlers.clear()

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.log') as temp_file:
        temp_filename = temp_file.name

    try:
        # Setup logging with file
        with patch('builtins.print') as mock_print:
            setup_dlml_logging(log_file=temp_filename)

        # Verify logger configuration
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 2  # stdout + file

        # Verify handlers
        handlers = logger.handlers
        stream_handler = next(
            h for h in handlers if isinstance(h, logging.StreamHandler)
        )
        file_handler = next(
            h for h in handlers if isinstance(h, logging.FileHandler)
        )

        # Verify formatters
        assert stream_handler.formatter._fmt == '%(message)s'
        assert (
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            in file_handler.formatter._fmt
        )

        # Verify print was called to announce log file
        mock_print.assert_called_once_with(f'Logging to file: {temp_filename}')

    finally:
        # Clean up
        logger.handlers.clear()
        temp_path = Path(temp_filename)
        if temp_path.exists():
            temp_path.unlink()


def test_setup_dlml_logging_with_timestamp_file() -> None:
    """Test setup_dlml_logging with automatic timestamp file creation."""
    # Clear any existing handlers
    logger = logging.getLogger('pelicun.dlml')
    logger.handlers.clear()

    with patch('builtins.print') as mock_print, patch(
        'pelicun.cli.datetime'
    ) as mock_datetime:
        # Mock datetime to return predictable timestamp
        mock_datetime.now.return_value.strftime.return_value = '2025-08-13_22-30-00'

        # Setup logging with True (auto-timestamp)
        setup_dlml_logging(log_file=True)

    # Verify logger configuration
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 2  # stdout + file

    # Verify print was called with timestamped filename
    expected_filename = 'dlml_update_2025-08-13_22-30-00.log'
    mock_print.assert_called_once_with(f'Logging to file: {expected_filename}')

    # Clean up
    logger.handlers.clear()
    # Clean up the created log file if it exists
    expected_path = Path(expected_filename)
    if expected_path.exists():
        expected_path.unlink()


def test_setup_dlml_logging_no_duplicate_handlers() -> None:
    """Test that setup_dlml_logging doesn't add duplicate handlers."""
    # Clear any existing handlers
    logger = logging.getLogger('pelicun.dlml')
    logger.handlers.clear()

    # Setup logging twice
    setup_dlml_logging(log_file=None)
    initial_handler_count = len(logger.handlers)

    setup_dlml_logging(log_file=None)
    final_handler_count = len(logger.handlers)

    # Should not add duplicate handlers
    assert initial_handler_count == final_handler_count == 1

    # Clean up
    logger.handlers.clear()


def test_setup_dlml_logging_file_creation() -> None:
    """Test that file logging actually creates and writes to the log file."""
    # Clear any existing handlers
    logger = logging.getLogger('pelicun.dlml')
    logger.handlers.clear()

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.log') as temp_file:
        temp_filename = temp_file.name

    try:
        # Setup logging with file
        setup_dlml_logging(log_file=temp_filename)

        # Log a test message
        logger.info('Test message for file logging')

        # Verify file was created and contains the message
        temp_path = Path(temp_filename)
        assert temp_path.exists()

        with temp_path.open('r') as f:
            content = f.read()
            assert 'Test message for file logging' in content
            assert 'pelicun.dlml' in content
            assert 'INFO' in content

    finally:
        # Clean up
        logger.handlers.clear()
        if temp_path.exists():
            temp_path.unlink()


def test_cli_integration_with_logging() -> None:
    """Test CLI integration with logging setup."""
    import subprocess  # noqa: PLC0415, S404

    # Test that CLI help includes the --log option
    result = subprocess.run(  # noqa: S603
        ['pelicun', 'dlml', '--help'],  # noqa: S607
        capture_output=True,
        text=True,
        cwd='/Users/adamzs/Repos/PBE/pelicun',
        check=False,
    )

    assert result.returncode == 0
    assert '--log [LOGFILE]' in result.stdout
    assert 'Save detailed log to specified file' in result.stdout
    assert 'creates dlml_update_TIMESTAMP.log' in result.stdout

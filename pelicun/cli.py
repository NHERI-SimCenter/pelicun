# Copyright (c) 2018 Leland Stanford Junior University
# Copyright (c) 2018 The Regents of the University of California
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

"""Provides a command-line interface for Pelicun."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

from pelicun.tools.dlml import dlml_update
from pelicun.tools.regional_sim import regional_sim


def setup_dlml_logging(log_file: str | bool | None = None) -> None:
    """
    Configure logging for DLML operations.

    Parameters
    ----------
    log_file : str, optional
        Path to log file. If True, creates timestamped file. If None, no file logging.
    """
    logger = logging.getLogger('pelicun.dlml')

    # Only add handlers if none exist (avoid duplicates)
    if not logger.handlers:
        # Always add stdout handler for CLI operations
        stdout_handler = logging.StreamHandler()
        stdout_formatter = logging.Formatter('%(message)s')
        stdout_handler.setFormatter(stdout_formatter)
        logger.addHandler(stdout_handler)

        # Add file handler if requested
        if log_file:
            if log_file is True:  # --log without filename
                timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # noqa: DTZ005
                log_file = f'dlml_update_{timestamp}.log'

            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

            print(f'Logging to file: {log_file}')

        logger.setLevel(logging.INFO)


def main() -> None:
    """
    Provide main command-line interface for Pelicun.

    This function dispatches subcommands.

    """
    # Define comprehensive usage examples
    examples = """
Examples:
  Regional Simulation:
    pelicun regional_sim                          # Use default config file (inputRWHALE.json)
    pelicun regional_sim my_config.json           # Use custom config file
    pelicun regional_sim -n 4                     # Use 4 CPU cores with default config
    pelicun regional_sim my_config.json -n 8      # Use custom config with 8 CPU cores

  DLML Data Management:
    pelicun dlml update                           # Update to latest DLML version
    pelicun dlml update latest                    # Same as above (explicit)
    pelicun dlml update v1.2.0                   # Update to specific version
    pelicun dlml update "commit abc1234"          # Update to specific commit SHA
    pelicun dlml update --no-cache latest         # Force re-download without caching
    pelicun dlml update --no-cache v1.2.0         # Update to version without caching
    pelicun dlml update --no-cache "commit def567" # Update to commit without caching

  Getting Help:
    pelicun --help                                # Show this help message
    pelicun regional_sim --help                   # Show regional_sim specific help
    pelicun dlml --help                           # Show dlml specific help
"""

    # Main parser
    parser = argparse.ArgumentParser(
        description='Main command-line interface for Pelicun.',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest='subcommand', required=True, help='Available subcommands'
    )

    # Create the parser for the "regional_sim" subcommand
    parser_regional = subparsers.add_parser(
        'regional_sim', help='Perform a regional-scale disaster impact simulation.'
    )

    # Add the arguments specific to regional_sim
    parser_regional.add_argument(
        'config_file',
        nargs='?',
        default='inputRWHALE.json',
        help='Path to the input configuration JSON file. '
        "Defaults to 'inputRWHALE.json'.",
    )
    parser_regional.add_argument(
        '-n',
        '--num-cores',
        type=int,
        default=None,
        help='Number of CPU cores to use for parallel processing. '
        'Defaults to all available cores minus one.',
    )
    # Associate the regional_sim function with this subparser
    parser_regional.set_defaults(func=regional_sim)

    # Create the parser for the "dlml" subcommand
    parser_dlml = subparsers.add_parser(
        'dlml', help='Update DLML (Damage and Loss Model Library) data files.'
    )

    # Add the arguments specific to dlml
    parser_dlml.add_argument(
        'action',
        choices=['update'],
        help='Action to perform. Currently only "update" is supported.',
    )
    parser_dlml.add_argument(
        'target',
        nargs='?',
        default='latest',
        help='Version tag (e.g., v1.2.0) or "latest" for the latest release. '
        'Use "commit <sha>" to specify a commit SHA.',
    )
    parser_dlml.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching to force re-download of all files.',
    )
    parser_dlml.add_argument(
        '--log',
        nargs='?',
        const=True,
        metavar='LOGFILE',
        help='Save detailed log to specified file. If no filename provided, '
        'creates dlml_update_TIMESTAMP.log in current directory.',
    )
    # Associate the dlml_update function with this subparser
    parser_dlml.set_defaults(func=dlml_update)

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Call the function associated with the chosen subcommand
    if args.subcommand == 'regional_sim':
        args.func(config_file=args.config_file, num_cores=args.num_cores)
    elif args.subcommand == 'dlml':
        # Setup logging for DLML operations
        setup_dlml_logging(log_file=args.log)

        # Handle dlml arguments
        use_cache = not args.no_cache
        if args.target.startswith('commit '):
            commit_sha = args.target.split(' ', 1)[1]
            args.func(commit=commit_sha, use_cache=use_cache)
        else:
            args.func(version=args.target, use_cache=use_cache)

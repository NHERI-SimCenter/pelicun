#
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
# John Vouvakis Manousakis


"""Constants, basic classes, and methods for pelicun."""

from __future__ import annotations

import argparse
import json
import pprint
import sys
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional, TypeVar, overload

import colorama
import numpy as np
import pandas as pd
from colorama import Fore, Style
from scipy.interpolate import interp1d  # type: ignore

from pelicun.pelicun_warnings import PelicunWarning

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from pelicun.assessment import AssessmentBase


colorama.init()
# set printing options
pp = pprint.PrettyPrinter(indent=2, width=80 - 24)

pd.options.display.max_rows = 20
pd.options.display.max_columns = None  # type: ignore
pd.options.display.expand_frame_repr = True
pd.options.display.width = 300

idx = pd.IndexSlice


T = TypeVar('T')


class Options:
    """
    Analysis options and logging configuration.

    Attributes
    ----------
    sampling_method: str
        Sampling method to use. Specified in the user's configuration
        dictionary, otherwise left as provided in the default configuration
        file (see settings/default_config.json in the pelicun source
        code). Can be any of ['LHS', 'LHS_midpoint',
        'MonteCarlo']. The default is 'LHS'.
    units_file: str
        Location of a user-specified units file, which should contain
        the names of supported units and their conversion factors (the
        value some quantity of a given unit needs to be multiplied to
        be expressed in the base units). Value specified in the user
        configuration dictionary. Pelicun comes with a set of default
        units which are always loaded (see
        `settings/default_units.json` in the pelicun source
        code). Units specified in the units_file overwrite the default
        units.
    demand_offset: dict
        Demand offsets are used in the process of mapping a component
        location to its associated EDP. This allows components that
        are sensitive to EDPs of different levels to be specified as
        present at the same location (e.g. think of desktop computer
        and suspended ceiling, both at the same story). Each
        component's offset value is specified in the component
        fragility database. This setting applies a supplemental global
        offset to specific EDP types. The value is specified in the
        user's configuration dictionary, otherwise left as provided in
        the default configuration file (see
        settings/default_config.json in the pelicun source code).
    nondir_multi_dict: dict
        Nondirectional components are sensitive to demands coming in
        any direction. Results are typically available in two
        orthogonal directions. FEMA P-58 suggests using the formula
        `max(dir_1, dir_2) * 1.2` to estimate the demand for such
        components. This parameter allows modifying the 1.2 multiplier
        with a user-specified value. The change can be applied to
        "ALL" EDPs, or for specific EDPs, such as "PFA", "PFV",
        etc. The value is specified in the user's configuration
        dictionary, otherwise left as provided in the default
        configuration file (see settings/default_config.json in the
        pelicun source code).
    rho_cost_time: float
        Specifies the correlation between the repair cost and repair
        time consequences. The value is specified in the user's
        configuration dictionary, otherwise left as provided in the
        default configuration file (see
        "RepairCostAndTimeCorrelation") (see
        settings/default_config.json in the pelicun source code).
    eco_scale: dict
        Controls how the effects of economies of scale are handled in
        the damaged component quantity aggregation for loss measure
        estimation. The dictionary is specified in the user's
        configuration dictionary, otherwise left as provided in the
        default configuration file (see settings/default_config.json
        in the pelicun source code).
    log: Logger
        Logger object. Configuration parameters coming from the user's
        configuration dictionary or the default configuration file
        control logging behavior. See Logger class.

    """

    __slots__ = [
        '_asmnt',
        '_rng',
        '_seed',
        'defaults',
        'demand_offset',
        'eco_scale',
        'eco_scale',
        'error_setup',
        'error_setup',
        'list_all_ds',
        'log',
        'log',
        'nondir_multi_dict',
        'rho_cost_time',
        'sampling_method',
        'units_file',
    ]

    def __init__(
        self,
        user_config_options: dict[str, Any] | None,
        assessment: AssessmentBase | None = None,
    ) -> None:
        """
        Initialize an Options object.

        Parameters
        ----------
        user_config_options: dict, Optional
            User-specified configuration dictionary. Any provided
            user_config_options override the defaults.
        assessment: AssessmentBase, Optional
            Assessment object that will be using this Options
            object. If it is not intended to use this Options object
            for an Assessment (e.g. defining an Options object for UQ
            use), this value should be None.

        """
        self._asmnt = assessment

        self.defaults: dict[str, Any] | None = None
        self.sampling_method: str | None = None
        self.list_all_ds: bool | None = None

        merged_config_options = merge_default_config(user_config_options)

        self.seed = merged_config_options['Seed']
        self.sampling_method = merged_config_options['Sampling']['SamplingMethod']
        self.list_all_ds = merged_config_options['ListAllDamageStates']

        self.units_file = merged_config_options['UnitsFile']

        self.demand_offset = merged_config_options['DemandOffset']
        self.nondir_multi_dict = merged_config_options['NonDirectionalMultipliers']
        self.rho_cost_time = merged_config_options['RepairCostAndTimeCorrelation']
        self.eco_scale = merged_config_options['EconomiesOfScale']

        self.error_setup = merged_config_options['ErrorSetup']

        # instantiate a Logger object with the finalized configuration
        self.log = Logger(
            merged_config_options['LogFile'],
            verbose=merged_config_options['Verbose'],
            log_show_ms=merged_config_options['LogShowMS'],
            print_log=merged_config_options['PrintLog'],
        )

    @property
    def seed(self) -> float | None:
        """
        Seed property.

        Returns
        -------
        float
            Seed value

        """
        return self._seed

    @seed.setter
    def seed(self, value: float) -> None:
        """Seed property setter."""
        self._seed = value
        self._rng = np.random.default_rng(self._seed)  # type: ignore

    @property
    def rng(self) -> np.random.Generator:
        """
        rng property.

        Returns
        -------
        Generator
            Random generator

        """
        return self._rng


# Define a module-level LoggerRegistry
class LoggerRegistry:
    """Registry to manage all logger instances."""

    _loggers: ClassVar[list[Logger]] = []

    # The @classmethod decorator allows this method to be called on
    # the class itself, rather than on instances. It interacts with
    # class-level data (like _loggers), enabling a single registry for
    # all Logger instances without needing an object of LoggerRegistry
    # itself.
    @classmethod
    def register(cls, logger: Logger) -> None:
        """Register a logger instance."""
        cls._loggers.append(logger)

    @classmethod
    def log_exception(
        cls,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
    ) -> None:
        """Log exceptions to all registered loggers."""
        message = (
            f"Unhandled exception occurred:"
            f"\n"
            f"{''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))}"
        )
        for logger in cls._loggers:
            logger.msg(message)

        # Also call the default excepthook to print the exception to
        # the console as is done by default.
        sys.__excepthook__(exc_type, exc_value, exc_traceback)


# Update sys.excepthook to log exceptions in all loggers
# https://docs.python.org/3/library/sys.html#sys.excepthook
sys.excepthook = LoggerRegistry.log_exception


class Logger:
    """Generate log files documenting execution events."""

    __slots__ = [
        'emitted',
        'log_div',
        'log_file',
        'log_show_ms',
        'log_time_format',
        'print_log',
        'spaces',
        'verbose',
        'warning_file',
        'warning_stack',
    ]

    def __init__(
        self,
        log_file: str | None,
        *,
        verbose: bool,
        log_show_ms: bool,
        print_log: bool,
    ) -> None:
        """
        Initialize a Logger object.

        Parameters
        ----------
        verbose: bool
            If True, the pelicun echoes more information throughout the
            assessment.  This can be useful for debugging purposes. The
            value is specified in the user's configuration dictionary,
            otherwise left as provided in the default configuration file
            (see settings/default_config.json in the pelicun source code).
        log_show_ms: bool
            If True, the timestamps in the log file are in microsecond
            precision. The value is specified in the user's configuration
            dictionary, otherwise left as provided in the default
            configuration file (see settings/default_config.json in the
            pelicun source code).
        log_file: str, optional
            If a value is provided, the log is written to that file. The
            value is specified in the user's configuration dictionary,
            otherwise left as provided in the default configuration file
            (see settings/default_config.json in the pelicun source code).
        print_log: bool
            If True, the log is also printed to standard output. The
            value is specified in the user's configuration dictionary,
            otherwise left as provided in the default configuration file
            (see settings/default_config.json in the pelicun source code).

        """
        self.verbose = verbose
        self.log_show_ms = bool(log_show_ms)

        if log_file is None:
            self.log_file = None
            self.warning_file = None
        else:
            path = Path(log_file)
            self.log_file = str(path.resolve())
            name, extension = split_file_name(self.log_file)
            self.warning_file = (
                path.parent / (name + '_warnings' + extension)
            ).resolve()
            with Path(self.log_file).open('w', encoding='utf-8') as f:
                f.write('')
            with Path(self.warning_file).open('w', encoding='utf-8') as f:
                f.write('')

        self.print_log = str2bool(print_log)
        self.warning_stack: list[str] = []
        self.emitted: set[str] = set()
        self.reset_log_strings()
        control_warnings()

        # Register the logger to the LoggerRegistry in order to
        # capture raised exceptions.
        LoggerRegistry.register(self)

    def reset_log_strings(self) -> None:
        """Populate the string-related attributes of the logger."""
        if self.log_show_ms:
            self.log_time_format = '%H:%M:%S:%f'
            # the length of the time string in the log file
            self.spaces = ' ' * 16
            # to have a total length of 80 with the time added
            self.log_div = '-' * (80 - 17)
        else:
            self.log_time_format = '%H:%M:%S'
            self.spaces = ' ' * 9
            self.log_div = '-' * (80 - 10)

    def msg(
        self,
        msg: str = '',
        *,
        prepend_timestamp: bool = True,
        prepend_blank_space: bool = True,
    ) -> None:
        """
        Write a message in the log file with the current time as prefix.

        The time is in ISO-8601 format, e.g. 2018-06-16T20:24:04Z

        Parameters
        ----------
        msg: string
            Message to print.
        prepend_timestamp: bool
            Controls whether a timestamp is placed before the message.
        prepend_blank_space: bool
            Controls whether blank space is placed before the message.

        """
        msg_lines = msg.split('\n')

        for msg_i, msg_line in enumerate(msg_lines):
            if prepend_timestamp and (msg_i == 0):
                formatted_msg = (
                    f'{datetime.now().strftime(self.log_time_format)} {msg_line}'  # noqa: DTZ005
                )
            elif prepend_timestamp or prepend_blank_space:
                formatted_msg = self.spaces + msg_line
            else:
                formatted_msg = msg_line

            if self.print_log:
                print(formatted_msg)  # noqa: T201

            if self.log_file is not None:
                with Path(self.log_file).open('a', encoding='utf-8') as f:
                    f.write('\n' + formatted_msg)

    def add_warning(self, msg: str) -> None:
        """
        Add a warning to the warning stack.

        Notes
        -----
        Warnings are only emitted when `emit_warnings` is called.

        Parameters
        ----------
        msg: str
            The warning message.

        """
        msg_lines = msg.split('\n')
        formatted_msg = '\n'
        for msg_line in msg_lines:
            formatted_msg += (
                self.spaces + Fore.RED + msg_line + Style.RESET_ALL + '\n'
            )
        if formatted_msg not in self.warning_stack:
            self.warning_stack.append(formatted_msg)

    def emit_warnings(self) -> None:
        """Issues all warnings and clears the warning stack."""
        for message in self.warning_stack:
            if message not in self.emitted:
                warnings.warn(message, PelicunWarning, stacklevel=3)
                if self.warning_file is not None:
                    with Path(self.warning_file).open('a', encoding='utf-8') as f:
                        f.write(
                            message.replace(Fore.RED, '')
                            .replace(Style.RESET_ALL, '')
                            .replace(self.spaces, '')
                        )

        self.emitted = self.emitted.union(set(self.warning_stack))
        self.warning_stack = []

    def warning(self, msg: str) -> None:
        """
        Add an emit a warning immediately.

        Parameters
        ----------
        msg: str
            Warning message

        """
        self.add_warning(msg)
        self.emit_warnings()

    def div(self, *, prepend_timestamp: bool = False) -> None:
        """Add a divider line in the log file."""
        msg = self.log_div if prepend_timestamp else '-' * 80
        self.msg(msg, prepend_timestamp=prepend_timestamp)

    def print_system_info(self) -> None:
        """Write system information in the log."""
        self.msg(
            'System Information:', prepend_timestamp=False, prepend_blank_space=False
        )
        start = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')  # noqa: DTZ005
        self.msg(
            f'local time zone: {datetime.now(timezone.utc).astimezone().tzinfo}\n'
            f'start time: {start}\n'
            f'python: {sys.version}\n'
            f'numpy: {np.__version__}\n'
            f'pandas: {pd.__version__}\n',
            prepend_timestamp=False,
        )


# get the absolute path of the pelicun directory
pelicun_path = Path(__file__).resolve().parent


def split_file_name(file_path: str) -> tuple[str, str]:
    """
    Separate a file name from the extension.

    Separates a file name from the extension accounting for the case
    where the file name itself contains periods.

    Parameters
    ----------
    file_path: str
        Original file path.

    Returns
    -------
    tuple
        name: str
            Name of the file.
        extension: str
            File extension.

    """
    path = Path(file_path)
    name = path.stem
    extension = path.suffix
    return name, extension


def control_warnings() -> None:
    """
    Turn warnings on/off.

        See also: `pelicun/pytest.ini`. Devs: make sure to update that
        file when addressing & eliminating warnings.

    """
    if not sys.warnoptions:
        # Here we specify *specific* warnings to ignore.
        # 'message' -- a regex that the warning message must match

        # Note: we ignore known warnings emitted from our dependencies
        # and plan to address them soon.

        warnings.filterwarnings(
            action='ignore', message='.*Use to_numeric without passing `errors`.*'
        )
        warnings.filterwarnings(
            action='ignore', message=".*errors='ignore' is deprecated.*"
        )
        warnings.filterwarnings(
            action='ignore',
            message='.*The previous implementation of stack is deprecated.*',
        )
        warnings.filterwarnings(
            action='ignore',
            message='.*Setting an item of incompatible dtype is deprecated.*',
        )
        warnings.filterwarnings(
            action='ignore',
            message='.*DataFrame.groupby with axis=1 is deprecated.*',
        )


def load_default_options() -> dict:
    """
    Load the default_config.json file to set options to default values.

    Returns
    -------
    dict
        Default options

    """
    with Path(pelicun_path / 'settings/default_config.json').open(
        encoding='utf-8'
    ) as f:
        default_config = json.load(f)

    return default_config['Options']


def update_vals(
    update_value: dict, primary: dict, update_path: str, primary_path: str
) -> None:
    """
    Transfer values between nested dictionaries.

    Updates the values of the `update` nested dictionary with
    those provided in the `primary` nested dictionary. If a key
    already exists in update, and does not map to another
    dictionary, the value is left unchanged.

    Parameters
    ----------
    update_value: dict
        Dictionary -which can contain nested dictionaries- to be
        updated based on the values of `primary`. New keys existing
        in `primary` are added to `update`. Values of which keys
        already exist in `primary` are left unchanged.
    primary: dict
        Dictionary -which can contain nested dictionaries- to
        be used to update the values of `update`.
    update_path: str
        Identifier for the update dictionary. Used to make error
        messages more meaningful.
    primary_path: str
        Identifier for the update dictionary. Used to make error
        messages more meaningful.

    Raises
    ------
    ValueError
      If primary[key] is dict but update[key] is not.
    ValueError
      If update[key] is dict but primary[key] is not.

    """
    # we go over the keys of `primary`
    for key in primary:  # noqa: PLC0206
        # if `primary[key]` is a dictionary:
        if isinstance(primary[key], dict):
            # if the same `key` does not exist in update,
            # we associate it with an empty dictionary.
            if key not in update_value:
                update_value[key] = {}
            # if it exists already, it should map to
            # a dictionary.
            elif not isinstance(update_value[key], dict):
                msg = (
                    f'{update_path}["{key}"] '
                    'should map to a dictionary. '
                    'The specified value is '
                    f'{update_path}["{key}"] = {update_value[key]}, but '
                    f'the default value is '
                    f'{primary_path}["{key}"] = {primary[key]}. '
                    f'Please revise {update_path}["{key}"].'
                )
                raise ValueError(msg)
            # With both being dictionaries, we use recursion.
            update_vals(
                update_value[key],
                primary[key],
                f'{update_path}["{key}"]',
                f'{primary_path}["{key}"]',
            )
        # if `primary[key]` is NOT a dictionary:
        elif key not in update_value:
            update_value[key] = primary[key]
        elif isinstance(update_value[key], dict):
            msg = (
                f'{update_path}["{key}"] '
                'should not map to a dictionary. '
                f'The specified value is '
                f'{update_path}["{key}"] = {update_value[key]}, but '
                f'the default value is '
                f'{primary_path}["{key}"] = {primary[key]}. '
                f'Please revise {update_path}["{key}"].'
            )
            raise ValueError(msg)


def merge_default_config(user_config: dict | None) -> dict:
    """
    Merge default config with user's options.

    Merge the user-specified config with the configuration defined in
    the default_config.json file. If the user-specified config does
    not include some option available in the default options, then the
    default option is used in the merged config.

    Parameters
    ----------
    user_config: dict
        User-specified configuration dictionary

    Returns
    -------
    dict
        Merged configuration dictionary

    """
    config = user_config  # start from the user's config
    default_config = load_default_options()

    if config is None:
        config = {}

    # We fill out the user's config with the values available in the
    # default config that were not set.
    # We use a recursive function to handle nesting.
    update_vals(config, default_config, 'user_settings', 'default_settings')

    return config


# https://stackoverflow.com/questions/52445559/
# how-can-i-type-hint-a-function-where-the-
# return-type-depends-on-the-input-type-o


@overload
def convert_to_SimpleIndex(
    data: pd.DataFrame, axis: int = 0, *, inplace: bool = False
) -> pd.DataFrame: ...


@overload
def convert_to_SimpleIndex(
    data: pd.Series, axis: int = 0, *, inplace: bool = False
) -> pd.Series: ...


def convert_to_SimpleIndex(  # noqa: N802
    data: pd.DataFrame | pd.Series, axis: int = 0, *, inplace: bool = False
) -> pd.DataFrame | pd.Series:
    """
    Convert the index of a DataFrame to a simple, one-level index.

    The target index uses standard SimCenter convention to identify
    different levels: a dash character ('-') is used to separate each
    level of the index.

    Parameters
    ----------
    data: DataFrame
        The DataFrame that will be modified.
    axis: int, optional, default:0
        Identifies if the index (0) or the columns (1) shall be
        edited.
    inplace: bool, optional, default:False
        If yes, the operation is performed directly on the input
        DataFrame and not on a copy of it.

    Returns
    -------
    DataFrame
        The modified DataFrame

    Raises
    ------
    ValueError
        When an invalid axis parameter is specified

    """
    if axis in {0, 1}:
        data_mod = data if inplace else data.copy()

        if axis == 0:
            # only perform this if there are multiple levels
            if data.index.nlevels > 1:
                simple_name = '-'.join(
                    [n if n is not None else '' for n in data.index.names]
                )
                simple_index = [
                    '-'.join([str(id_i) for id_i in idx]) for idx in data.index
                ]

                data_mod.index = pd.Index(simple_index, name=simple_name)
                data_mod.index.name = simple_name

        elif axis == 1:
            # only perform this if there are multiple levels
            if data.columns.nlevels > 1:
                simple_name = '-'.join(
                    [n if n is not None else '' for n in data.columns.names]
                )
                simple_index = [
                    '-'.join([str(id_i) for id_i in idx]) for idx in data.columns
                ]

                data_mod.columns = pd.Index(simple_index, name=simple_name)
                data_mod.columns.name = simple_name

    else:
        msg = f'Invalid axis parameter: {axis}'
        raise ValueError(msg)

    return data_mod


@overload
def convert_to_MultiIndex(
    data: pd.DataFrame, axis: int = 0, *, inplace: bool = False
) -> pd.DataFrame: ...


@overload
def convert_to_MultiIndex(
    data: pd.Series, axis: int = 0, *, inplace: bool = False
) -> pd.Series: ...


def convert_to_MultiIndex(  # noqa: N802
    data: pd.DataFrame | pd.Series, axis: int = 0, *, inplace: bool = False
) -> pd.DataFrame | pd.Series:
    """
    Convert the index of a DataFrame to a MultiIndex.

    We assume that the index uses standard SimCenter convention to
    identify different levels: a dash character ('-') is expected to
    separate each level of the index.

    Parameters
    ----------
    data: DataFrame
        The DataFrame that will be modified.
    axis: int, optional, default:0
        Identifies if the index (0) or the columns (1) shall be
        edited.
    inplace: bool, optional, default:False
        If yes, the operation is performed directly on the input
        DataFrame and not on a copy of it.

    Returns
    -------
    DataFrame
        The modified DataFrame.

    Raises
    ------
    ValueError
        If an invalid axis is specified.

    """
    # check if the requested axis is already a MultiIndex
    if ((axis == 0) and (isinstance(data.index, pd.MultiIndex))) or (
        (axis == 1) and (isinstance(data.columns, pd.MultiIndex))
    ):
        # if yes, return the data unchanged
        return data

    if axis == 0:
        index_labels = [str(label).split('-') for label in data.index]

    elif axis == 1:
        index_labels = [str(label).split('-') for label in data.columns]

    else:
        msg = f'Invalid axis parameter: {axis}'
        raise ValueError(msg)

    max_lbl_len = np.max([len(labels) for labels in index_labels])

    for l_i, labels in enumerate(index_labels):
        if len(labels) != max_lbl_len:
            labels += [''] * (max_lbl_len - len(labels))  # noqa: PLW2901
            index_labels[l_i] = labels

    index_labels_np = np.array(index_labels)

    if index_labels_np.shape[1] > 1:
        data_mod = data if inplace else data.copy()

        if axis == 0:
            data_mod.index = pd.MultiIndex.from_arrays(index_labels_np.T)

        else:
            data_mod.columns = pd.MultiIndex.from_arrays(index_labels_np.T)

        return data_mod

    return data


def convert_dtypes(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to a numeric datatype whenever possible.

    The function replaces None with NA otherwise columns containing
    None would continue to have the `object` type.

    Parameters
    ----------
    dataframe: DataFrame
        The DataFrame that will be modified.

    Returns
    -------
    DataFrame
        The modified DataFrame.

    """
    with (
        pd.option_context('future.no_silent_downcasting', True),  # noqa: FBT003
        pd.option_context('mode.copy_on_write', True),  # noqa: FBT003
    ):
        dataframe = dataframe.fillna(value=np.nan).infer_objects()
    # note: `axis=0` applies the function to the columns
    # note: ignoring errors is a bad idea and should never be done. In
    # this case, however, that's not what we do, despite the name of
    # this parameter. We simply don't convert the dtype of columns
    # that cannot be interpreted as numeric. That's what
    # `errors='ignore'` does.
    # See:
    # https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html
    return dataframe.apply(
        lambda x: pd.to_numeric(x, errors='ignore'),  # type:ignore
        axis=0,
    )


def show_matrix(
    data: np.ndarray | pd.DataFrame, *, use_describe: bool = False
) -> None:
    """
    Print a matrix in a nice way using a DataFrame.

    Parameters
    ----------
    data: array-like
        The matrix data to display. Can be any array-like structure
        that pandas can convert to a DataFrame.
    use_describe: bool, default: False
        If True, provides a descriptive statistical summary of the
        matrix including specified percentiles.
        If False, simply prints the matrix as is.

    """
    if use_describe:
        pp.pprint(
            pd.DataFrame(data).describe(percentiles=[0.01, 0.1, 0.5, 0.9, 0.99])
        )
    else:
        pp.pprint(pd.DataFrame(data))


def multiply_factor_multiple_levels(
    df: pd.DataFrame,
    conditions: dict,
    factor: float,
    axis: int = 0,
    *,
    raise_missing: bool = True,
) -> None:
    """
    Multiply a value to selected rows, in place.

    Multiplies a value to selected rows of a DataFrame that is indexed
    with a hierarchical index (pd.MultiIndex). The change is done in
    place.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to be modified.
    conditions: dict
        A dictionary mapping level names with a single value. Only the
        rows where the index levels have the provided values will be
        affected. The dictionary can be empty, in which case all rows
        will be affected, or contain only some levels and values, in
        which case only the matching rows will be affected.
    factor: float
        Scaling factor to use.
    axis: int
        With 0 the condition is checked against the DataFrame's index,
        otherwise with 1 it is checked against the DataFrame's
        columns.
    raise_missing: bool
        Raise an error if no rows are matching the given conditions.

    Raises
    ------
    ValueError
        If the provided `axis` values is not either 0 or 1.
    ValueError
        If there are no rows matching the conditions and raise_missing
        is True.

    """
    if axis == 0:
        idx_to_use = df.index
    elif axis == 1:
        idx_to_use = df.columns
    else:
        msg = f'Invalid axis: `{axis}`'
        raise ValueError(msg)

    mask = pd.Series(data=True, index=idx_to_use)

    # Apply each condition to update the mask
    for level, value in conditions.items():
        mask &= idx_to_use.get_level_values(level) == value

    if np.all(mask == False) and raise_missing:  # noqa: E712
        msg = f'No rows found matching the conditions: `{conditions}`'
        raise ValueError(msg)

    if axis == 0:
        df.iloc[mask.to_numpy()] *= factor
    else:
        df.iloc[:, mask.to_numpy()] *= factor


def _warning(
    message: str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: Any = None,  # noqa: ARG001, ANN401
    line: Any = None,  # noqa: ARG001, ANN401
) -> None:
    """
    Display warnings in a custom format.

    Custom warning function to format and print warnings more
    attractively. This function modifies how warning messages are
    displayed, emphasizing the file path and line number from where
    the warning originated.

    Parameters
    ----------
    message: str
        The warning message to be displayed.
    category: Warning
        The category of the warning (unused, but required for
        compatibility with standard warning signature).
    filename: str
        The path of the file from which the warning is issued. The
        function simplifies the path for display.
    lineno: int
        The line number in the file at which the warning is issued.
    file: file-like object, optional
        The target file object to write the warning to (unused, but
        required for compatibility with standard warning signature).
    line: str, optional
        Line of code causing the warning (unused, but required for
        compatibility with standard warning signature).

    """
    # pylint:disable = unused-argument
    if category != PelicunWarning:
        if '\\' in filename:
            file_path = filename.split('\\')
        elif '/' in filename:
            file_path = filename.split('/')
        else:
            file_path = None

        python_file = '/'.join(file_path[-3:]) if file_path is not None else filename
        print(f'WARNING in {python_file} at line {lineno}\n{message}\n')  # noqa: T201
    else:
        print(message)  # noqa: T201


warnings.showwarning = _warning  # type: ignore


def describe(
    data: pd.DataFrame | pd.Series | np.ndarray,
    percentiles: tuple[float, ...] = (
        0.001,
        0.023,
        0.10,
        0.159,
        0.5,
        0.841,
        0.90,
        0.977,
        0.999,
    ),
) -> pd.DataFrame:
    """
    Extend descriptive statistics.

    Provides extended descriptive statistics for given data, including
    percentiles and log standard deviation for applicable columns.
    This function accepts both pandas Series and DataFrame objects
    directly, or any array-like structure which can be converted to
    them. It calculates common descriptive statistics and optionally
    adds log standard deviation for columns where all values are
    positive.

    Parameters
    ----------
    data: pd.Series, pd.DataFrame, or array-like
        The data to describe. If array-like, it is converted to a
        DataFrame or Series before analysis.
    percentiles: tuple of float, optional
        Specific percentiles to include in the output. Default
        includes an extensive range tailored to provide a detailed
        summary.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the descriptive statistics of the input
        data, transposed so that each descriptive statistic is a row.

    """
    if isinstance(data, np.ndarray):
        vals = data

        if vals.ndim == 1:
            data = pd.Series(vals, name=0)
        else:
            cols = np.arange(vals.shape[1])
            data = pd.DataFrame(vals, columns=cols)

    # convert Series to a DataFrame
    if isinstance(data, pd.Series):
        data = pd.DataFrame(data)

    desc = pd.DataFrame(data.describe(list(percentiles)).T)

    # add log standard deviation to the stats
    desc.insert(3, 'log_std', np.nan)
    desc = desc.T

    for col in desc.columns:
        if np.min(data[col]) > 0.0:
            desc.loc['log_std', col] = np.std(np.log(data[col]), ddof=1)

    return desc


def str2bool(v: str | bool) -> bool:  # noqa: FBT001
    """
    Convert a string representation of truth to boolean True or False.

    This function is designed to convert string inputs that represent
    boolean values into actual Python boolean types. It handles
    typical representations of truthiness and falsiness, and is case
    insensitive.

    Parameters
    ----------
    v: str or bool
        The value to convert into a boolean. This can be a boolean
        itself (in which case it is simply returned) or a string that
        is expected to represent a boolean value.

    Returns
    -------
    bool
        The boolean value corresponding to the input.

    Raises
    ------
    argparse.ArgumentTypeError
        If `v` is a string that does not correspond to a boolean
        value, an error is raised indicating that a boolean value was
        expected.

    """
    # courtesy of Maxim @ Stackoverflow

    if isinstance(v, bool):
        return v
    if v.lower() in {'yes', 'true', 'True', 't', 'y', '1'}:
        return True
    if v.lower() in {'no', 'false', 'False', 'f', 'n', '0'}:
        return False
    msg = 'Boolean value expected.'
    raise argparse.ArgumentTypeError(msg)


def float_or_None(string: str) -> float | None:  # noqa: N802
    """
    Convert strings to float or None.

    Parameters
    ----------
    string: str
        A string

    Returns
    -------
    float or None
        A float, if the given string can be converted to a
        float. Otherwise, it returns None

    """
    try:
        return float(string)
    except ValueError:
        return None


def int_or_None(string: str) -> int | None:  # noqa: N802
    """
    Convert strings to int or None.

    Parameters
    ----------
    string: str
        A string

    Returns
    -------
    int or None
        An int, if the given string can be converted to an
        int. Otherwise, it returns None

    """
    try:
        return int(string)
    except ValueError:
        return None


def check_if_str_is_na(string: Any) -> bool:  # noqa: ANN401
    """
    Check if the provided string can be interpreted as N/A.

    Parameters
    ----------
    string: object
            The string to evaluate

    Returns
    -------
    bool
        The evaluation result. Yes, if the string is considered N/A.
    """
    na_vals = {
        '',
        'N/A',
        '-1.#QNAN',
        'null',
        'None',
        '<NA>',
        'nan',
        '-NaN',
        '1.#IND',
        'NaN',
        '#NA',
        '1.#QNAN',
        'NULL',
        '-nan',
        '#N/A',
        '#N/A N/A',
        'n/a',
        '-1.#IND',
        'NA',
    }
    # obtained from Pandas' internal STR_NA_VALUES variable.

    return isinstance(string, str) and string in na_vals


def with_parsed_str_na_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify string values interpretable as N/A.

    Given a dataframe, this function identifies values that have
    string type and can be interpreted as N/A, and replaces them with
    actual NA's.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe to process

    Returns
    -------
    pd.DataFrame
        The dataframe with proper N/A values.
    """
    # Replace string NA values with actual NaNs
    return df.apply(
        lambda col: col.map(lambda x: np.nan if check_if_str_is_na(x) else x)
    )


def dedupe_index(dataframe: pd.DataFrame, dtype: type = str) -> pd.DataFrame:
    """
    Add a `uid` level to the index.

    Modifies the index of a DataFrame to ensure all index elements are
    unique by adding an extra level.  Assumes that the DataFrame's
    original index is a MultiIndex with specified names. A unique
    identifier ('uid') is added as an additional index level based on
    the cumulative count of occurrences of the original index
    combinations.

    Parameters
    ----------
    dataframe: pd.DataFrame
        The DataFrame whose index is to be modified. It must have a
        MultiIndex.
    dtype: type, optional
        The data type for the new index level 'uid'. Defaults to str.

    Returns
    -------
    dataframe: pd.DataFrame
        The original dataframe with an additional `uid` level at the
        index.

    """
    inames = dataframe.index.names
    dataframe = dataframe.reset_index()
    dataframe['uid'] = (dataframe.groupby([*inames]).cumcount()).astype(dtype)
    dataframe = dataframe.set_index([*inames, 'uid'])
    return dataframe.sort_index()


# Input specs

EDP_to_demand_type = {
    # Drifts
    'Story Drift Ratio': 'PID',
    'Peak Interstory Drift Ratio': 'PID',
    'Roof Drift Ratio': 'PRD',
    'Peak Roof Drift Ratio': 'PRD',
    'Damageable Wall Drift': 'DWD',
    'Racking Drift Ratio': 'RDR',
    'Mega Drift Ratio': 'PMD',
    'Residual Drift Ratio': 'RID',
    'Residual Interstory Drift Ratio': 'RID',
    'Peak Effective Drift Ratio': 'EDR',
    # Floor response
    'Peak Floor Acceleration': 'PFA',
    'Peak Floor Velocity': 'PFV',
    'Peak Floor Displacement': 'PFD',
    # Component response
    'Peak Link Rotation Angle': 'LR',
    'Peak Link Beam Chord Rotation': 'LBR',
    # Wind Intensity
    'Peak Gust Wind Speed': 'PWS',
    # Wind Demands
    'Peak Wind Force': 'PWF',
    'Peak Internal Force': 'PIF',
    'Peak Line Force': 'PLF',
    'Peak Wind Pressure': 'PWP',
    # Inundation Intensity
    'Peak Inundation Height': 'PIH',
    # Shaking Intensity
    'Peak Ground Acceleration': 'PGA',
    'Peak Ground Velocity': 'PGV',
    'Spectral Acceleration': 'SA',
    'Spectral Velocity': 'SV',
    'Spectral Displacement': 'SD',
    'Peak Spectral Acceleration': 'SA',
    'Peak Spectral Velocity': 'SV',
    'Peak Spectral Displacement': 'SD',
    'Permanent Ground Deformation': 'PGD',
    # Placeholder for advanced calculations
    'One': 'ONE',
}


def dict_raise_on_duplicates(ordered_pairs: list[tuple]) -> dict:
    """
    Construct a dictionary from a list of key-value pairs.

    Constructs a dictionary from a list of key-value pairs, raising an
    exception if duplicate keys are found.
    This function ensures that no two pairs have the same key. It is
    particularly useful when parsing JSON-like data where unique keys
    are expected but not enforced by standard parsing methods.

    Parameters
    ----------
    ordered_pairs: list of tuples
        A list of tuples, each containing a key and a value. Keys are
        expected to be unique across the list.

    Returns
    -------
    dict
        A dictionary constructed from the ordered_pairs without any
        duplicates.

    Raises
    ------
    ValueError
        If a duplicate key is found in the input list, a ValueError is
        raised with a message indicating the duplicate key.

    Examples
    --------
    >>> dict_raise_on_duplicates(
    ...     [("key1", "value1"), ("key2", "value2"), ("key1", "value3")]
    ... )
    ValueError: duplicate key: key1

    Notes
    -----
    This implementation is useful for contexts in which data integrity
    is crucial and key uniqueness must be ensured.

    """
    d = {}
    for k, v in ordered_pairs:
        if k in d:
            msg = f'duplicate key: {k}'
            raise ValueError(msg)
        d[k] = v
    return d


def parse_units(  # noqa: C901
    custom_file: str | None = None, *, preserve_categories: bool = False
) -> dict:
    """
    Parse the unit conversion factor JSON file and return a dictionary.

    Parameters
    ----------
    custom_file: str, optional
        If a custom file is provided, only the units specified in the
        custom file are used.
    preserve_categories: bool, optional
        If True, maintains the original data types of category
        values from the JSON file. If False, converts all values
        to floats and flattens the dictionary structure, ensuring
        that each unit name is globally unique across categories.


    Returns
    -------
    dict
        A dictionary where keys are unit names and values are
        their corresponding conversion factors. If
        `preserve_categories` is True, the dictionary may maintain
        its original nested structure based on the JSON file. If
        `preserve_categories` is False, the dictionary is flattened
        to have globally unique unit names.

    """

    def get_contents(file_path: Path, *, preserve_categories: bool = False) -> dict:  # noqa: C901
        """
        Map unit names to conversion factors.

        Parses a unit conversion factors JSON file and returns a
        dictionary mapping unit names to conversion factors.

        This function allows the use of a custom JSON file for
        defining unit conversion factors or defaults to a predefined
        file. It ensures that each unit name is unique and that all
        conversion factors are float values. Additionally, it supports
        the option to preserve the original data types of category
        values from the JSON.

        Parameters
        ----------
        file_path: str
            The file path to a JSON file containing unit conversion
            factors. If not provided, a default file is used.
        preserve_categories: bool, optional
            If True, maintains the original data types of category
            values from the JSON file. If False, converts all values
            to floats and flattens the dictionary structure, ensuring
            that each unit name is globally unique across categories.

        Returns
        -------
        dict
            A dictionary where keys are unit names and values are
            their corresponding conversion factors. If
            `preserve_categories` is True, the dictionary may maintain
            its original nested structure based on the JSON file.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        ValueError
            If a unit name is duplicated or other JSON structure issues are present.
        TypeError
            If a conversion factor is not a float.
        TypeError
            If any value that needs to be converted to float cannot be
            converted.

        Examples
        --------
        >>> parse_units('custom_units.json')
        { 'm': 1.0, 'cm': 0.01, 'mm': 0.001 }

        >>> parse_units('custom_units.json', preserve_categories=True)
        { 'Length': {'m': 1.0, 'cm': 0.01, 'mm': 0.001} }

        """
        try:
            with Path(file_path).open(encoding='utf-8') as f:
                dictionary = json.load(f, object_pairs_hook=dict_raise_on_duplicates)
        except FileNotFoundError as exc:
            msg = f'{file_path} was not found.'
            raise FileNotFoundError(msg) from exc
        except json.decoder.JSONDecodeError as exc:
            msg = f'{file_path} is not a valid JSON file.'
            raise ValueError(msg) from exc
        for category_dict in list(dictionary.values()):
            # ensure all first-level keys point to a dictionary
            if not isinstance(category_dict, dict):
                msg = (
                    f'{file_path} contains first-level keys '
                    "that don't point to a dictionary"
                )
                raise TypeError(msg)
            # convert values to float
            try:
                for key, val in category_dict.items():
                    category_dict[key] = float(val)
            except (ValueError, TypeError) as exc:
                msg = (
                    f'Unit {key} has a value of {val} '
                    'which cannot be interpreted as a float'
                )
                raise type(exc)(msg) from exc

        if preserve_categories:
            return dictionary

        flattened = {}
        for category in dictionary:
            for unit_name, factor in dictionary[category].items():
                if unit_name in flattened:
                    msg = f'{unit_name} defined twice in {file_path}.'
                    raise ValueError(msg)
                flattened[unit_name] = factor

        return flattened

    if custom_file:
        return get_contents(
            Path(custom_file), preserve_categories=preserve_categories
        )

    return get_contents(
        pelicun_path / 'settings/default_units.json',
        preserve_categories=preserve_categories,
    )


def convert_units(  # noqa: C901
    values: float | list[float] | np.ndarray,
    unit: str,
    to_unit: str,
    category: str | None = None,
) -> float | list[float] | np.ndarray:
    """
    Convert numeric values between different units.

    Supports conversion within a specified category of units and
    automatically infers the category if not explicitly provided. It
    maintains the type of the input in the output.

    Parameters
    ----------
    values: (float | list[float] | np.ndarray)
      The numeric value(s) to convert.
    unit: (str)
      The current unit of the values.
    to_unit: (str)
      The target unit to convert the values into.
    category: (Optional[str])
      The category of the units (e.g., 'length', 'pressure'). If not
      provided, the category will be inferred based on the provided
      units.

    Returns
    -------
    float or list[float] or np.ndarray
      The converted value(s) in the target unit, in the same data type
      as the input values.

    Raises
    ------
    TypeError
      If the input `values` are not of type float, list, or
      np.ndarray.
    ValueError
      If the `unit`, `to_unit`, or `category` is unknown or if `unit`
      and `to_unit` are not in the same category.

    """
    if isinstance(values, (float, list)):
        vals = np.atleast_1d(values)
    elif isinstance(values, np.ndarray):
        vals = values
    else:
        msg = 'Invalid input type for `values`'
        raise TypeError(msg)

    # load default units
    all_units = parse_units(preserve_categories=True)

    # if a category is given use it, otherwise try to determine it
    if category:
        if category not in all_units:
            msg = f'Unknown category: `{category}`'
            raise ValueError(msg)
        units = all_units[category]
        for unt in unit, to_unit:
            if unt not in units:
                msg = f'Unknown unit: `{unt}`'
                raise ValueError(msg)
    else:
        unit_category: str | None = None
        for key in all_units:
            units = all_units[key]
            if unit in units:
                unit_category = key
                break
        if not unit_category:
            msg = f'Unknown unit `{unit}`'
            raise ValueError(msg)
        units = all_units[unit_category]
        if to_unit not in units:
            msg = (
                f'`{unit}` is a `{unit_category}` unit, but `{to_unit}` '
                f'is not specified in that category.'
            )
            raise ValueError(msg)

    # convert units
    from_factor = units[unit]
    to_factor = units[to_unit]
    new_values = vals * float(from_factor) / float(to_factor)

    # return the results in the same type as that of the provided
    # values
    if isinstance(values, float):
        return new_values[0]
    if isinstance(values, list):
        return new_values.tolist()
    return new_values


def stringterpolation(
    arguments: str,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Linear interpolation from strings.

    Turns a string of specially formatted arguments into a multilinear
    interpolating function.

    Parameters
    ----------
    arguments: str
        String of arguments containing Y values and X values,
        separated by a pipe symbol (`|`). Individual values are
        separated by commas (`,`). Example:
        arguments = 'y1,y2,y3|x1,x2,x3'

    Returns
    -------
    Callable
        A callable interpolating function

    """
    split = arguments.split('|')
    x_vals = split[1].split(',')
    y_vals = split[0].split(',')
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)

    return interp1d(x=x, y=y, kind='linear')


def invert_mapping(original_dict: dict) -> dict:
    """
    Inverts a dictionary mapping from key to list of values.

    Parameters
    ----------
    original_dict: dict
        Dictionary with values that are lists of hashable items.

    Returns
    -------
    dict
        New dictionary where each item in the original value lists
        becomes a key and the original key becomes the corresponding
        value.

    Raises
    ------
    ValueError
        If any value in the original dictionary's value lists appears
        more than once.

    """
    inverted_dict = {}
    for key, value_list in original_dict.items():
        for value in value_list:
            if value in inverted_dict:
                msg = 'Cannot invert mapping with duplicate values.'
                raise ValueError(msg)
            inverted_dict[value] = key
    return inverted_dict


def get(
    d: dict | None,
    path: str,
    default: Any | None = None,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """
    Path-like dictionary value retrieval.

    Retrieves a value from a nested dictionary using a path with '/'
    as the separator.

    Parameters
    ----------
    d: dict
        The dictionary to search.
    path: str
        The path to the desired value, with keys separated by '/'.
    default: Any, optional
        The value to return if the path is not found. Defaults to
        None.

    Returns
    -------
    Any
        The value found at the specified path, or the default value if
        the path is not found.

    Examples
    --------
    >>> config = {
    ...     "DL": {
    ...         "Outputs": {
    ...             "Format": {
    ...                 "JSON": "desired_value"
    ...             }
    ...         }
    ...     }
    ... }
    >>> get(config, '/DL/Outputs/Format/JSON', default='default_value')
    'desired_value'
    >>> get(config, '/DL/Outputs/Format/XML', default='default_value')
    'default_value'

    """
    if d is None:
        return default
    keys = path.strip('/').split('/')
    current_dict = d
    try:
        for key in keys:
            current_dict = current_dict[key]
        return current_dict  # noqa: TRY300
    except (KeyError, TypeError):
        return default


def update(
    d: dict[str, Any],
    path: str,
    value: Any,  # noqa: ANN401
    *,
    only_if_empty_or_none: bool = False,
) -> None:
    """
    Set a value in a nested dictionary using a path with '/' as the separator.

    Parameters
    ----------
    d: dict
        The dictionary to update.
    path: str
        The path to the desired value, with keys separated by '/'.
    value: Any
        The value to set at the specified path.
    only_if_empty_or_none: bool, optional
        If True, only update the value if it is None or an empty
        dictionary. Defaults to False.

    Examples
    --------
    >>> d = {}
    >>> update(d, 'x/y/z', 1)
    >>> d
    {'x': {'y': {'z': 1}}}

    >>> update(d, 'x/y/z', 2, only_if_empty_or_none=True)
    >>> d
    {'x': {'y': {'z': 1}}}  # value remains 1 since it is not empty or None

    >>> update(d, 'x/y/z', 2)
    >>> d
    {'x': {'y': {'z': 2}}}  # value is updated to 2

    """
    keys = path.strip('/').split('/')
    current_dict = d
    for key in keys[:-1]:
        if key not in current_dict or not isinstance(current_dict[key], dict):
            current_dict[key] = {}
        current_dict = current_dict[key]
    if only_if_empty_or_none:
        if is_unspecified(current_dict, keys[-1]):
            current_dict[keys[-1]] = value
    else:
        current_dict[keys[-1]] = value


def is_unspecified(d: dict[str, Any], path: str) -> bool:
    """
    Check if something is specified.

    Checks if a value in a nested dictionary is either non-existent,
    None, NaN, or an empty dictionary or list.

    Parameters
    ----------
    d: dict
        The dictionary to search.
    path: str
        The path to the desired value, with keys separated by '/'.

    Returns
    -------
    bool
        True if the value is non-existent, None, or an empty
        dictionary or list. False otherwise.

    Examples
    --------
    >>> config = {
    ...     "DL": {
    ...         "Outputs": {
    ...             "Format": {
    ...                 "JSON": "desired_value",
    ...                 "EmptyDict": {}
    ...             }
    ...         }
    ...     }
    ... }
    >>> is_unspecified(config, '/DL/Outputs/Format/JSON')
    False
    >>> is_unspecified(config, '/DL/Outputs/Format/XML')
    True
    >>> is_unspecified(config, '/DL/Outputs/Format/EmptyDict')
    True

    """
    value = get(d, path, default=None)
    if value is None:
        return True
    if pd.isna(value):
        return True
    if value == {}:
        return True
    return value == []


def is_specified(d: dict[str, Any], path: str) -> bool:
    """
    Opposite of `is_unspecified()`.

    Parameters
    ----------
    d: dict
        The dictionary to search.
    path: str
        The path to the desired value, with keys separated by '/'.

    Returns
    -------
    bool
        True if the value is specified, False otherwise.

    """
    return not is_unspecified(d, path)


def ensure_value(value: T | None) -> T:
    """
    Ensure a variable is not None.

    This function checks that the provided variable is not None. It is
    used to assist with type hinting by avoiding repetitive `assert
    value is not None` statements throughout the code.

    Parameters
    ----------
    value : Optional[T]
        The variable to check, which can be of any type or None.

    Returns
    -------
    T
        The same variable, guaranteed to be non-None.

    Raises
    ------
    TypeError
        If the provided variable is None.

    """
    if value is None:
        raise TypeError
    return value

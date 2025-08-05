import argparse
from pelicun.tools.regional_sim import regional_sim

def main():
    """
    Main command-line interface for Pelicun.
    This function dispatches subcommands.
    """
    # Main parser
    parser = argparse.ArgumentParser(
        description="Main command-line interface for Pelicun."
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True,
                                       help="Available subcommands")

    # Create the parser for the "regional_sim" subcommand
    parser_regional = subparsers.add_parser(
        "regional_sim",
        help="Perform a regional-scale disaster impact simulation."
    )

    # Add the arguments specific to regional_sim
    parser_regional.add_argument(
        "config_file",
        nargs='?',
        default="inputRWHALE.json",
        help="Path to the input configuration JSON file. "
             "Defaults to 'inputRWHALE.json'."
    )
    parser_regional.add_argument(
        "-n", "--num-cores",
        type=int,
        default=None,
        help="Number of CPU cores to use for parallel processing. "
             "Defaults to all available cores minus one."
    )
    # Associate the regional_sim function with this subparser
    parser_regional.set_defaults(func=regional_sim)

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Call the function associated with the chosen subcommand
    if args.subcommand == "regional_sim":
        args.func(config_file=args.config_file, num_cores=args.num_cores)
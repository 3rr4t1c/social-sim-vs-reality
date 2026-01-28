#!/usr/bin/env python3
"""
Social Media Simulator Validation
==================================
Entry point for conversion and comparative analysis.

Usage:
    python analyze.py                        # Full analysis
    python analyze.py --convert input/ out/  # Convert simulator data
    python analyze.py --plots temporal       # Only specific plots
    python analyze.py --no-cache             # Ignore cache
    python analyze.py -v                     # Verbose logging

Examples:
    # Full analysis with settings from settings.py
    python analyze.py

    # Convert simulator output
    python analyze.py --convert data_conversion/synthetic/input_data/ data/synthetic/new_sim/

    # Generate only specific plots
    python analyze.py --plots temporal cascade

    # List available plots
    python analyze.py --list-plots
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Social Media Simulator Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py                          Full analysis
  python analyze.py --convert raw/ data/sim/ Convert simulator output
  python analyze.py --plots temporal cascade Generate only specific plots
  python analyze.py --list-plots             Show available plots
        """,
    )

    # Conversion mode
    parser.add_argument(
        "--convert",
        nargs=2,
        metavar=("INPUT", "OUTPUT"),
        help="Convert simulator data: --convert input_dir output_dir",
    )

    # Plot selection
    parser.add_argument(
        "--plots",
        nargs="+",
        metavar="PLOT",
        help="Generate only specified plots (temporal, cascade, cdf, iet, response_time, quality)",
    )
    parser.add_argument(
        "--list-plots",
        action="store_true",
        help="Show list of available plots",
    )

    # Options
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore cache and recalculate everything",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable detailed logging",
    )

    args = parser.parse_args()

    # Import here for fast startup when only --help is used
    from src.pipeline import run_validation, run_conversion, list_plots

    if args.list_plots:
        list_plots()
        return 0

    if args.convert:
        converted = run_conversion(
            args.convert[0],
            args.convert[1],
            verbose=args.verbose,
        )
        return 0 if converted > 0 else 1

    run_validation(
        plots=args.plots,
        use_cache=not args.no_cache,
        verbose=args.verbose,
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nFatal error: {e}")
        raise

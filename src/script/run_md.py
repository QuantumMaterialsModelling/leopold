"""
MD script for the Leopoldo package

creation: 2025-05-26 11:26:07
author:   Luca Leoni
contact:  luca.leoni12@unibo.it
"""

# ==== IMPORTS ==== #

# Argument Parser
from argparse import ArgumentParser, Namespace


# ==== FUNCTIONS ==== #

# ==== PARSER ==== #


def parse_args() -> Namespace:
    parser = ArgumentParser("run_md")

    return parser.parse_args()


# ==== MAIN ==== #


def main():
    args = parse_args()


if __name__ == "__main__":
    main()

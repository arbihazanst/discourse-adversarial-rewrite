#!/usr/bin/env python3
"""Run PPLX then Clef, merge prediction logs, and calculate benchmark metrics."""
import sys
from classify import execute, install_stop_handler, parser


def main(argv=None):
    args = parser(single=False).parse_args(argv)
    return execute(args, ['pplx', 'clef'], analyze=True)


if __name__ == '__main__':
    install_stop_handler()
    sys.exit(main())

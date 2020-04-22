#!/usr/bin/env python

import argparse

from vaelib.pytorch_categorical_flow import main
from vaelib.pytorch_categorical_flow import construct_parser

if __name__ == "__main__":
    parser = construct_parser()
    args = parser.parse_args()
    main(args)
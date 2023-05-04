#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq_cli.eval_lm import cli_main
'''
diy setting
'''
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
'''
diy setting
'''

if __name__ == '__main__':
    cli_main()

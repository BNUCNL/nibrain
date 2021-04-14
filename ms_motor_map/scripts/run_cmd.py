"""
    run command
"""

import os, subprocess, argparse

def run_cmd(args):
    

if __name__ == '__main__':

    # initialize argparse
    parser = argparse.ArgumentParser()

    """
       required parameters 
    """
    parser.add_argument("projectdir", help="base dir contains all project files.")
    parser.add_argument("cmd", help="cmd")

    """
        optinal parameters 
    """
    parser.add_argument("-i", "--subject", action="store_true", help="if choose, validator will initialize a new projectdir.")
    parser.add_argument("-c", "--session", action="store_true", help="if choose, validator will create required folder if it is not exist.")
    parser.add_argument("-o", "--run", help="dir contains original data, if not None, validator will create a soft link to original data dir. It should an be absolute path.")

    args = parser.parse_args()

    # CNLS validation
    run_cmd(args)
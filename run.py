import argparse
import os
import sys

from config import Config
from concurrent.futures import ProcessPoolExecutor, as_completed
from memory import Memory

PATH = os.path.dirname(os.path.dirname(__file__))

if PATH not in sys.path:
    sys.path.append(PATH)

commands = ['self', 'opt', 'play']

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', help='what to do', choices=commands)
    return parser

parser = create_parser()
args = parser.parse_args()

config = Config()

if args.cmd == 'self':
    from process.play import SelfPlayProcess

    process = SelfPlayProcess(config, Memory(config))

    # Create a pool of workers and execute the self plays
    with ProcessPoolExecutor(max_workers=16) as executor:
        _ = [executor.submit(process.play, log=(i == 0)) for i in range(config.workers)]

elif args.cmd == 'opt':
    from process import optimize
    optimize.start(config)
elif args.cmd == 'play':
    pass
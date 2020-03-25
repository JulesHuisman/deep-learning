import argparse
import os
import sys

from config import Config
from concurrent.futures import ProcessPoolExecutor, as_completed
from memory import Memory

PATH = os.path.dirname(os.path.dirname(__file__))

if PATH not in sys.path:
    sys.path.append(PATH)

commands = ['self', 'opt', 'eval']

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('process', help='What process to run', choices=commands)
    return parser

parser = create_parser()
args = parser.parse_args()

config = Config()
memory = Memory(config)

if args.process == 'self':
    from process.play import SelfPlayProcess

    process = SelfPlayProcess(config, memory)

    # Create a pool of workers and execute the self plays
    with ProcessPoolExecutor(max_workers=16) as executor:
        [executor.submit(process.play, log=(i == 0)) for i in range(config.workers)]

elif args.process == 'opt':
    from process.optimize import OptimizeProcess

    process = OptimizeProcess(config, memory)
    process.optimize()

elif args.process == 'eval':
    from process.evaluate import EvaluateProcess

    process = EvaluateProcess(config)
    process.evaluate()
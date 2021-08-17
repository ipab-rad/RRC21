"""
Planning for the stage 2 of RRC: rearange dice
"""
import os

from trifinger_simulation.tasks import rearrange_dice as task

from pddlstream.algorithms.meta import solve, create_parser
from pddlstream.algorithms.search import solve
from pddlstream.utils import read


def read_pddl(filename: str):
    dir = os.path.dirname(os.path.abspath(__file__))
    return read(os.path.join(dir, filename))



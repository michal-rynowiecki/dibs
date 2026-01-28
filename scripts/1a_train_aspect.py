import sys
import os

project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(1, project_root)

from src.training.utils import read_input
from model import Model
from utils import ArgumentParser

args = ArgumentParser().parser.parse_args()
model = Model(args)

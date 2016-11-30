from show_attend_tell_model import Model as show_attend_tell
from utils import ArgumentParser

args = ArgumentParser().parser.parse_args()
model = show_attend_tell(args, mode="train")
model.build()

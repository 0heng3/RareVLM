"""Independent annotation-supervised object SFT with optional TextVQA replay.

Frozen model/input helpers are reused explicitly; no preference/reference loss
or earlier posttraining source is modified by this study.
"""
import sys

sys.dont_write_bytecode = True

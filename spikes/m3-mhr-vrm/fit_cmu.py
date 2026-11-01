"""Fit MHR to one CMU clip and print residuals: uv run python fit_cmu.py SUBJECT CLIP"""

import sys

from mtcore.cmu_fit import fit

if __name__ == "__main__":
    fit(int(sys.argv[1]), int(sys.argv[2]))

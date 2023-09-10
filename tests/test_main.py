import sys
import pandas as pd
import matplotlib.pyplot as plt

# sys.path.append("/workspaces/fj49_week2_ds")
from sorc.main import mean, median, mode, std, viz, reader

spotify = reader()
mean = mean()
median = median()
mode = mode()
std = std()
viz = viz()


def test_mean():
    assert mean == int(spotify["duration_ms"].mean())


def test_median():
    assert median == int(spotify["duration_ms"].median())


def test_mode():
    assert mode == int(spotify["duration_ms"].mode())


def test_std():
    assert std == int(spotify["duration_ms"].std())


def test_viz():
    assert viz is not None  # asserts that viz is not empty

import pandas as pd
import matplotlib.pyplot as plt


# reading the data
def reader():
    spotify = pd.read_csv("playlist_2010to2022.csv")
    return spotify


# basic stats
def mean():
    mean_duration = int(spotify["duration_ms"].mean())
    return mean_duration


def median():
    median_duration = int(spotify["duration_ms"].median())
    return median_duration


def mode():
    mode_duration = int(spotify["duration_ms"].mode())
    return mode_duration


def std():
    std_duration = int(spotify["duration_ms"].std())
    return std_duration


# making a plot
def viz():
    value_counts = spotify["artist_name"].value_counts()
    top_10_value_counts = value_counts.head(10)

    plt.figure(figsize=(20, 12))
    plt.bar(top_10_value_counts.index, top_10_value_counts.values)
    # Add labels and a title to the plot
    plt.xlabel("Top Artists")
    plt.ylabel("Number of top tracks")
    plt.title("Which artists had the most top tracks in the last few years?")
    plt.show()


# def sanity_add(x, y):
#     return x + y


# if __name__ == "__main__":
#     a = 1
#     b = 2

#     print(sanity_add(a, b))

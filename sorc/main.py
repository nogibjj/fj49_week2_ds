import pandas as pd
import matplotlib.pyplot as plt


# reading the data
def reader():
    df = pd.read_csv("spotify.csv")
    return df


spotify = reader()
print(spotify)


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
    x = plt.figure(figsize=(20, 12))
    x = plt.bar(top_10_value_counts.index, top_10_value_counts.values)
    # Add labels and a title to the plot
    x = plt.xlabel("Top Artists")
    x = plt.ylabel("Number of top tracks")
    x = plt.title("Which artists had the most top tracks in the last few years?")
    plt.show()
    return x


# def sanity_add(x, y):
#     return x + y


# if __name__ == "__main__":
#     a = 1
#     b = 2

#     print(sanity_add(a, b))

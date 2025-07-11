from collections import defaultdict
import matplotlib.pyplot as plt

%matplotlib inline

"""
This histogram will plot all the offsets for the top song. If the song recognition app is performing well, the offsets should be clustered around a single area and not
extend too far. It accepts the dictionary from tally_fingerprints of (song_id, offset):matches.
"""
def plot_offset_histogram(tallies, top_song):
    offsets = []
    for (song_id, offset), count in tallies.items():
        if song_id == top_song:
            offsets.extend([offset] * count)

    plt.figure(figsize = (16,8))
    plt.hist(offsets, bins = 30)
    plt.xlabel("Offset")
    plt.ylabel("Match Count")
    plt.title(f"Offset Histogram for {top_song}")
    plt.tight_layout()
    plt.savefig(f"{top_song}_offset.png")
    plt.close()


"""
This function plots the top songs against their total matches. If the matches of top song of few songs is much different from the other songs, it could indicate a more
confident or well-performing algorithm. It accepts the dictionary from tally_fingerprints of (song_id, offset):matches.
"""
def plot_top_matches(scores, output_path = "top_matches.png"):
    combined_scores = {}
    for key in scores:
        song_id = key[0]
        count = scores[key]

        if song_id in combined_scores:
            combined_scores[song_id] += count
        else:
            combined_scores[song_id] = count

    sorted_items = sorted(scores.items(), key = lambda item: item[1], reverse = True)
    top_songs = []
    top_scores = []
    for song_id, score in sorted_items:
        top_songs.append(song_id)
        top_scores.append(score)

    plt.figure(figsize = (20, 10))
    plt.plt(top_songs, top_scores, marker = "o")
    plt.xlabel("Song ID")
    plt.ylabel("Match Score")
    plt.title("Top Match Score by Song")
    plt.xsticks(rotation = 45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


"""
This function is for if we digitially add noise to our platform! It tracks match scores as more noise is added. This might not be used.
It would need to take in a list of noise levels and their corresponding top song match scores, so this plot might not be feasible.
"""
def noise_plot(noise_levels, scores):
    plt.figure(figsize = (8,5))
    plt.plot(noise_levels, scores, market = "o", linestyle = "-")
    plt.xlabel("Noise Level")
    plt.ylabel("Match Score (Correct Song)")
    plt.title("Noise Robustness of Song Recognition")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("noise_plot.png")
    plt.close()

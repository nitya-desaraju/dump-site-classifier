import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from typing import Tuple

%matplotlib notebook

def search(fanout_m, database):
    matches = {}
    for (fm, fn, dt), tm in fanout_m:
        if (fm, fn, dt) in database:
            for song, time in database[(fm, fn, dt)]:
                
                t_offset = int(time - tm)
                if str((song, t_offset)) in matches:
                    matches[str((song, t_offset))] += 1
                else:
                    matches[str((song, t_offset))] = 1
    songs = list(matches.keys())
    counts = list(matches.values())
    plt.figure(figsize=(6, 4))
    plt.bar(songs, counts)
    plt.xlabel("Song")
    plt.ylabel("Likelihood (Number of Matches)")
    plt.title("Song vs Likelihood")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    best_song = max(matches, key=matches.get)
    print(best_song)


            
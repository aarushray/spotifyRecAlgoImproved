
import pandas as pd
import numpy as np

df = pd.read_csv("data1.csv")

tracks = ["Creep", "Smells Like Teen Spirit"]

lis = ["danceability",
                "energy", 
                "valence",
                "acousticness",
                "speechiness",
                "instrumentalness",
                "liveness",
                "tempo",
                ]

result = (
    df[df["track_name"].isin(tracks)]
    .drop_duplicates(subset="track_name", keep="first")
    [lis]
)




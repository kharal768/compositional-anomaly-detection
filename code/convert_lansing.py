"""Convert the Lansing Woods census exported from R to the npz file used by the scripts.

In R (spatstat.data installed):
    library(spatstat.data); data(lansing)
    write.csv(data.frame(x = lansing$x, y = lansing$y,
                         mark = as.integer(lansing$marks) - 1), "lansing.csv", row.names = FALSE)
Then place lansing.csv in data/ and run:  python code/convert_lansing.py
"""
import numpy as np, pandas as pd
d = pd.read_csv("data/lansing.csv")
levels = ["blackoak", "hickory", "maple", "misc", "redoak", "whiteoak"]   # order of spatstat's marks
np.savez("data/lansing.npz", x=d["x"].to_numpy(float), y=d["y"].to_numpy(float),
         m=d["mark"].to_numpy(int), levels=np.array(levels))
print("wrote data/lansing.npz with", len(d), "trees")

#%%
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import sys
sys.path.insert(0, "..")
from DatasetBuilder import build_df
import pandas as pd

#df=build_df(root_file="../Nano.root")
df=pd.read_parquet("../CCTk_match.parquet")
#%%
pt=df["CryClu_pt"].to_numpy()
y=df["CryClu_label"].to_numpy()

#%%
plt.hist(pt[y==0],bins=200,range=(0,100),histtype="step",label="Background")
plt.hist(pt[y==1],bins=200,range=(0,100),histtype="step",label="Signal")
plt.plot([2,2],[1,1e7])
plt.ylim(1,1e7)
plt.yscale("log")
plt.xlabel("CryClu_pt")
plt.legend()

#%%
genpt=df["CryClu_genPt"].to_numpy()
plt.hist2d(pt[genpt > 0], genpt[genpt > 0],bins=40,range=((0,20),(0,50)))
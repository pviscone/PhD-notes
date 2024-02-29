# %%
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import importlib
import utils
import pandas as pd

importlib.reload(utils)
from utils import flat, delta_r, label_builder, snapshot_wrapper, evIdx

hep.style.use("CMS")
NanoAODSchema.warn_missing_crossrefs = False
"""
Tk
#!mvaQual is always 0
#! hovere is always 0
CC
#! pucorrpt is always 0
#! bremstrength is always 0
#!electronWP is always 0
#!photonWP is always 0
#!stage2effMatch is always 0
#!.looseL1TkMatchWP is always 0
"""
features = [
    "CryClu_standaloneWP",
    "CryClu_showerShape",
    "CryClu_isolation",
    "Tk_hitPattern",
    "Tk_nStubs",
    "Tk_chi2Bend",
    "Tk_chi2RPhi",
    "Tk_chi2RZ",
    "Tk_vz",
    "CCTk_dR",
    "CCTk_dPt",
    "CCTk_PtRatio",
    "CryClu_pt",
    "CryClu_label",
    "CryClu_genPt",
    "CryClu_evIdx",
]

dRcut_ccGen = 0.1
dRcut_ccTk = 0.1
save = False
#save="CCTk_match.parquet"


# %%
def build_df(features=features, root_file="Nano.root", save=False):
    events = NanoEventsFactory.from_root(
        {root_file: "Events"},
        schemaclass=NanoAODSchema,
    ).events()


    GenEle = events.GenEl
    GenEle = GenEle[np.abs(GenEle.eta) < 1.48]
    inAcceptanceMask = ak.num(GenEle) > 0
    events = events[inAcceptanceMask]

    GenEle = GenEle[inAcceptanceMask].compute()
    CryClu = events.CaloCryCluGCT.compute()
    Tk = events.DecTkBarrel.compute()


    CryClu["GenIdx"], GenEle["CryCluIdx"] = label_builder(
        ak.ArrayBuilder(), ak.ArrayBuilder(), CryClu, GenEle, dRcut=dRcut_ccGen
    )

    Tk["CryCluIdx"], CryClu["TkIdx"] = label_builder(
        ak.ArrayBuilder(), ak.ArrayBuilder(), Tk, CryClu, dRcut=dRcut_ccTk
    )

    CryClu = CryClu[~ak.is_none(CryClu.TkIdx, axis=1)]
    CryClu["label"] = ak.values_astype(ak.fill_none(CryClu.GenIdx, -1) > -1, int)
    CryClu["showerShape"] = CryClu.e2x5 / CryClu.e5x5
    Tk = Tk[CryClu.TkIdx]
    CryClu["genPt"] = ak.fill_none(GenEle.pt[CryClu.GenIdx], -1)
    CryClu["evIdx"]=evIdx(ak.ArrayBuilder(), CryClu)

    CryClu = ak.flatten(CryClu)
    Tk = ak.flatten(Tk)

    CCTk = ak.Array([{}] * len(CryClu))
    CCTk["dR"] = delta_r(
        ak.to_numpy(CryClu.phi, allow_missing=False),
        ak.to_numpy(Tk.caloPhi, allow_missing=False),
        ak.to_numpy(CryClu.eta, allow_missing=False),
        ak.to_numpy(Tk.caloEta, allow_missing=False),
    )

    CCTk["dPt"] = Tk.pt - CryClu.pt
    CCTk["PtRatio"]=Tk.pt/CryClu.pt

    res = np.empty((len(Tk), 1))
    for feature in features:
        obj, feature = feature.split("_")
        arr=eval(f"ak.to_numpy(ak.to_numpy({obj}.{feature},allow_missing=False))")
        arr = np.expand_dims(arr, axis=1)
        res = np.concatenate((res, arr), axis=1)

    res = res[:, 1:]


    df = pd.DataFrame(res, columns=features)
    if save:
        df.to_parquet(save)

    return df


if __name__ == "__main__":
    df=build_df(save=save)




# %% #! Plot input
'''
import matplotlib.lines as mlines
import corner

l = [lab.split("_")[-1] for lab in df.columns]

blue_line = mlines.Line2D([], [], color="tab:blue", label="Signal")
red_line = mlines.Line2D([], [], color="tab:orange", label="Bkg")
fig = corner.corner(
    df[df["CryClu_label"] == 1].drop(columns=["CryClu_label", "CryClu_genPt"]),
    levels=(0.5, 0.9, 0.99),
    color="tab:blue",
    scale_hist=False,
    plot_density=False,
    bins=15,
    labels=l,
)
corner.corner(
    df[df["CryClu_label"] == 0].drop(columns=["CryClu_label", "CryClu_genPt"]),
    levels=(0.5, 0.9, 0.99),
    fig=fig,
    color="tab:orange",
    scale_hist=False,
    bins=15,
    plot_density=False,
    labels=l,
)

plt.legend(
    handles=[blue_line, red_line],
    bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
    loc=4,
)
'''
# %%
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import importlib
import sys
import copy

sys.path.append("../")
import utils

importlib.reload(utils)
from utils import flat, label_builder  # noqa: E402


hep.style.use("CMS")
NanoAODSchema.warn_missing_crossrefs = False

events = NanoEventsFactory.from_root(
    {"../DoubleElectrons131X.root": "Events"},
    schemaclass=NanoAODSchema,
).events()

GenEle = events.GenEl
GenEle = GenEle[np.abs(GenEle.eta) < 1.48]
inAcceptanceMask = ak.num(GenEle) > 0

GenEle = GenEle[inAcceptanceMask]
CryClu = events.CaloCryCluGCT[inAcceptanceMask]
OldCryClu=events.CaloCryClu[inAcceptanceMask]
Tk = events.DecTkBarrel[inAcceptanceMask]
TkEle = events.TkEleL2[inAcceptanceMask]

GenEleCalo = copy.copy(GenEle)
GenEleCalo["phi"]=GenEleCalo.calophi
GenEleCalo["eta"]=GenEleCalo.caloeta
TkCalo = copy.copy(Tk)
TkCalo["phi"]=TkCalo.caloPhi
TkCalo["eta"]=TkCalo.caloEta


GenEle = GenEle.compute()
GenEleCalo = GenEleCalo.compute()
CryClu = CryClu.compute()
OldCryClu=OldCryClu.compute()
Tk = Tk.compute()
TkCalo = TkCalo.compute()
TkEle = TkEle.compute()


GenEle = ak.with_name(GenEle, "PtEtaPhiMLorentzVector")
GenEleCalo = ak.with_name(GenEleCalo, "PtEtaPhiMLorentzVector")
CryClu = ak.with_name(CryClu, "PtEtaPhiMLorentzVector")
OldCryClu = ak.with_name(OldCryClu, "PtEtaPhiMLorentzVector")
Tk = ak.with_name(Tk, "PtEtaPhiMLorentzVector")
TkCalo = ak.with_name(TkCalo, "PtEtaPhiMLorentzVector")
TkEle = ak.with_name(TkEle, "PtEtaPhiMLorentzVector")


#%%
def nearest_dr(obj1,obj2):
    _, dr = obj1.nearest(obj2, return_metric=True)
    return dr


cc_gen_dr = nearest_dr(GenEleCalo,CryClu)
old_cc_gen_dr = nearest_dr(GenEleCalo,OldCryClu)
tk_gen_dr = nearest_dr(GenEle,Tk)
tkele_gen_dr = nearest_dr(GenEleCalo,TkEle)
cc_tk_dr=nearest_dr(CryClu,TkCalo)
old_cc_tk_dr=nearest_dr(OldCryClu,TkCalo)

#%%
def dr_plots(dr, bins=np.linspace(0, 3,80), ax=None, **kwargs):
    if not ax:
        _, ax = plt.subplots()

    ax.hist(flat(dr), bins=bins, histtype="step", linewidth=2,density=True,**kwargs)
    ax.grid()
    #ax.set_xlabel("$\Delta R$")
    ax.set_yscale("log")
    #ax.set_ylim(0.005,1e2)
    ax.legend(fontsize=15)
    return ax

_,ax=plt.subplots(2,2,figsize=(10,10))

ax[0, 0] = dr_plots(old_cc_gen_dr, label="OldCryClu-GenEle", ax=ax[0, 0])
ax[0, 0] = dr_plots(cc_gen_dr, label="CryClu-GenEle", ax=ax[0, 0])

ax[0, 1] = dr_plots(tk_gen_dr, label="Tk-GenEle", ax=ax[0, 1])
ax[1, 0] = dr_plots(tkele_gen_dr, label="TkEle-GenEle", ax=ax[1, 0])
ax[1, 1] = dr_plots(cc_tk_dr, label="CryClu-Tk", ax=ax[1, 1])
ax[1, 1] = dr_plots(old_cc_tk_dr, label="OldCryClu-Tk", ax=ax[1, 1])
ax[1,1].set_xlabel("$\Delta R$")
ax[0,0].set_ylabel("Density")

# %%


CryClu["GenIdx"], GenEle["CryCluIdx"] = label_builder(
    ak.ArrayBuilder(), ak.ArrayBuilder(), CryClu, GenEleCalo, dRcut=0.1
)

OldCryClu["GenIdx"], GenEle["OldCryCluIdx"] = label_builder(
    ak.ArrayBuilder(), ak.ArrayBuilder(), OldCryClu, GenEleCalo, dRcut=0.1
)

Tk["OldCryCluIdx"], OldCryClu["TkIdx"] = label_builder(
    ak.ArrayBuilder(), ak.ArrayBuilder(), TkCalo, OldCryClu, dRcut=0.2
)

Tk["CryCluIdx"], CryClu["TkIdx"] = label_builder(
    ak.ArrayBuilder(), ak.ArrayBuilder(), TkCalo, CryClu, dRcut=0.2
)

Tk["GenIdx"], GenEle["TkIdx"] = label_builder(
    ak.ArrayBuilder(), ak.ArrayBuilder(), Tk, GenEle, dRcut=0.1
)

TkEle["GenIdx"], GenEle["TkEleIdx"] = label_builder(
    ak.ArrayBuilder(), ak.ArrayBuilder(), TkEle, GenEleCalo, dRcut=0.1
)


# %%
def matching_eff(Gen, Obj, bins=np.linspace(0,100,50), obj2=None, obj_ref=None, ax=None,**kwargs):
    if obj_ref is None:
        genHist, genEdges = np.histogram(flat(Gen.pt), bins=bins)
    else:
        genHist, genEdges = np.histogram(flat(obj_ref), bins=bins)

    centers = (genEdges[:-1] + genEdges[1:]) / 2

    if obj2 is None:
        obj_genHist, _ = np.histogram(flat(Gen[Obj.GenIdx].pt), bins=genEdges)
    else:
        obj_genHist, _ = np.histogram(flat(obj2.pt), bins=genEdges)
    if ax is None:
        _, ax = plt.subplots()


    ax.step(centers, obj_genHist / genHist, where="mid", **kwargs)

    ax.legend(fontsize=25)

    ax.grid(visible=True)
    ax.set_ylim(0,1.1)
    return ax

_,ax=plt.subplots(1,2,figsize=(20,10))
ax[0]=matching_eff(GenEleCalo, CryClu, bins=np.linspace(0,100,50),label="CryClu-Gen",ax=ax[0])
ax[0]=matching_eff(GenEleCalo, OldCryClu, bins=np.linspace(0,100,50),label="OldCryClu-Gen",ax=ax[0])

ax[0].set_ylabel("Efficiency")

ax[0]=matching_eff(GenEle, Tk, bins=np.linspace(0,100,50),label="Tk-Gen",ax=ax[0])







ax[1] = matching_eff(
    GenEle,
    None,
    obj2=GenEle[
        CryClu[
            (
                ak.fill_none(Tk[CryClu.TkIdx].pt > -1, False)
                & ak.fill_none(GenEle[CryClu.GenIdx].pt > -1, False)
            )
        ].GenIdx
    ],
    bins=np.linspace(0, 100, 50),
    label="Total\nCryClu-Tk-Gen",
    ax=ax[1],
)

ax[1] = matching_eff(
    GenEle,
    None,
    obj2=GenEle[
        OldCryClu[
            (
                ak.fill_none(Tk[OldCryClu.TkIdx].pt > -1, False)
                & ak.fill_none(GenEle[OldCryClu.GenIdx].pt > -1, False)
            )
        ].GenIdx
    ],
    bins=np.linspace(0, 100, 50),
    label="Total\nOldCryClu-Tk-Gen",
    ax=ax[1],
)

ax[1] = matching_eff(
    GenEleCalo, TkEle, bins=np.linspace(0, 100, 50), label="TkEle-Gen", ax=ax[1],color="green",linestyle='--'
)


ax[1].set_xlabel("GenEle $p_T$ [GeV]")

# %%
"""
def plot_efficiencies(Gen, CryClu, Tk, bins=50,ax=None):
    tk_gen = Gen[Tk.GenIdx]
    cc_gen = Gen[CryClu.GenIdx]
    tk_cc_gen = Gen[Tk[CryClu[Gen.CryCluIdx].TkIdx].GenIdx]

    tkele_gen = Gen[TkEle.GenIdx]

    genHist, genEdges = np.histogram(Gen.pt, bins=bins, range=(0, 100))
    centers = (genEdges[:-1] + genEdges[1:]) / 2

    tk_genHist, _ = np.histogram(ak.flatten(tk_gen).pt, bins=genEdges)
    cc_genHist, _ = np.histogram(ak.flatten(cc_gen).pt, bins=genEdges)
    tk_cc_genHist, _ = np.histogram(ak.flatten(tk_cc_gen).pt, bins=genEdges)
    tkele_genHist, _ = np.histogram(ak.flatten(tkele_gen).pt, bins=genEdges)

    if ax is None:
        _, ax = plt.subplots()
        ax.set_xlabel("GenEle $p_T$ [GeV]")

    ax.step(centers, cc_genHist / genHist, where="mid", label="CryClu-Gen")
    ax.step(centers, tk_genHist / genHist, where="mid", label="Tk-Gen")
    ax.step(centers, tk_cc_genHist / genHist, where="mid", label="Tk-CryClu-Gen")
    ax.step(centers, tkele_genHist / genHist, where="mid", label="TkEle-Gen",)
    ax.legend()
    ax.set_ylabel("Efficiency")
    ax.grid()
    return ax


fig,ax=plt.subplots(2,2,figsize=(10,10))

plot_efficiencies(GenEle, CryClu, Tk, bins=100,ax=ax[0,0])

# quello che faccio è per ogni cryclu, matcho un tk. Devovedificare che entrambi matchano lo stesso genEle
#!Plot matched CryClu pt vs gen pt
ax[0, 1].hist2d(
    flat(CryClu[GenEle.CryCluIdx].pt),
    flat(GenEle[CryClu.GenIdx].pt),
    bins=(75, 75),
)

#!Plot matched Tk pt vs Gen pt
ax[1,0].hist2d(
    flat(GenEle[Tk.GenIdx].pt),
    flat(Tk[GenEle.TkIdx].pt),
    bins=(75, 75),
    range=((0, 100), (0, 100)),
)

matched_cc = CryClu[GenEle.CryCluIdx]
cc_tkidx=ak.drop_none(matched_cc.TkIdx)

ax[1,1].hist2d(
    flat(CryClu[Tk[cc_tkidx].CryCluIdx].pt),
    flat(Tk[cc_tkidx].pt),
    bins=(75, 75),
    range=((0, 100), (0, 100)),
)
ax[1,1].yaxis.tick_right()
ax[0, 1].yaxis.tick_right()
ax[0, 1].yaxis.set_label_position("right")
ax[1, 1].yaxis.set_label_position("right")
ax[0, 1].set_ylabel("GenEle $p_T$")
ax[1, 1].set_ylabel("Tk $p_T$")
ax[1,0].set_ylabel("Tk $p_T$")

ax[1,0].set_xlabel("GenEle $p_T$")
ax[1,1].set_xlabel("Matched CryClu $p_T$")
hep.cms.text("Phase-2 Simulation",ax=ax[0,0],fontsize=20)
#%%
#!dR plot
dR_cc_gen = delta_r(
    flat(GenEle.calophi[~ak.is_none(GenEle.CryCluIdx, axis=1)]),
    flat(CryClu.phi[GenEle.CryCluIdx]),
    flat(GenEle.caloeta[~ak.is_none(GenEle.CryCluIdx, axis=1)]),
    flat(CryClu.eta[GenEle.CryCluIdx]),
)

dR_tk_gen = delta_r(
    flat(GenEle.calophi[~ak.is_none(GenEle.TkIdx, axis=1)]),
    flat(Tk.phi[GenEle.TkIdx]),
    flat(GenEle.caloeta[~ak.is_none(GenEle.TkIdx, axis=1)]),
    flat(Tk.eta[GenEle.TkIdx]),
)

cc = CryClu[GenEle.CryCluIdx]
cc=cc[~ak.is_none(cc.TkIdx,axis=1)]
tk=Tk[cc.TkIdx]
dR_tk_cc = delta_r(
    flat(cc.phi),
    flat(tk.phi),
    flat(cc.eta),
    flat(tk.eta),
)

plt.hist([dR_cc_gen, dR_tk_gen, dR_tk_cc], bins=20, histtype="step", label=["CryClu-Gen", "Tk-Gen", "Tk-CryClu"],linewidth=2)
plt.legend()
plt.yscale("log")
plt.xlabel("$\Delta R$")
hep.cms.text("Phase-2 Simulation",fontsize=20)

#%%#!BINNED dr  CC-Tk in gen pt



pt_bins = np.linspace(0, 50, 7)

stack=[]
labels=[]
for edge1,edge2 in zip(pt_bins[:-1],pt_bins[1:]):
    mask = ((GenEle.pt) > edge1) & ((GenEle.pt) < edge2)
    gen=GenEle[mask]
    cc=CryClu[gen.CryCluIdx]
    cc=cc[~ak.is_none(cc.TkIdx,axis=1)]
    cc=cc[~ak.is_none(cc.GenIdx,axis=1)]
    tk=Tk[cc.TkIdx]
    dR_tk_cc = delta_r(
        flat(cc.phi),
        flat(tk.phi),
        flat(cc.eta),
        flat(tk.eta),
    )
    stack.append(dR_tk_cc)
    labels.append(f"{edge1:.1f} < Gen $p_T$ < {edge2:.1f}")

fig,axes=plt.subplots(len(stack),1,sharex=True,figsize=(10,30))
plt.subplots_adjust(hspace=0)
for idx,ax in enumerate(axes):
    ax.hist(stack[idx], bins=20, label=labels[idx],linewidth=2,histtype="step",range=(0,0.2))
    ax.legend()
    ax.set_yscale("log")
    ax.grid()

    if idx!=len(stack)-1:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels([0]+[f"{i:.2f}" for i in np.linspace(0,0.2,5)])
hep.cms.text("Phase-2 Simulation",fontsize=20,ax=axes[0])
axes[-1].set_xlabel("$\Delta R$")


 """

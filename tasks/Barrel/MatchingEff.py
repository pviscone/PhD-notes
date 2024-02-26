# %%
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import numba as nb


hep.style.use("CMS")
NanoAODSchema.warn_missing_crossrefs = False

events = NanoEventsFactory.from_root(
    {"Nano.root": "Events"},
    schemaclass=NanoAODSchema,
).events()


#!REMOVE THIS, debug only
# events=events[:2]
def flat(akarr):
    return ak.to_numpy(ak.drop_none(ak.flatten(akarr)))


@nb.jit(nopython=True)
def delta_phi(phi1, phi2):
    return (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi


@nb.jit(nopython=True)
def delta_r(phi1, phi2, eta1, eta2):
    return np.sqrt((eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2)

def snapshot_wrapper(func):
    def wrapper(*args, **kwargs):
        res12, res21 = func(*args, **kwargs)
        return res12.snapshot(), res21.snapshot()
    return wrapper

@snapshot_wrapper
@nb.jit(nopython=True)
def label_builder(builder12, builder21, Obj, ObjToLoopOn, dRcut):
    for event_idx in range(len(Obj)):
        builder12.begin_list()
        builder21.begin_list()
        obj = Obj[event_idx]
        res12 = np.ones(len(obj), dtype=nb.int32) * -1
        res21 = np.ones(len(ObjToLoopOn[event_idx]), dtype=nb.int32) * -1
        for idx, ele in enumerate(ObjToLoopOn[event_idx]):
            dr = delta_r(
                np.array(ele.phi),
                np.array(obj.phi),
                np.array(ele.eta),
                np.array(obj.eta),
            )
            # For each objToLoopOn, match the objwith dR<0.2 and the hightest pt
            if np.sum(dr < dRcut) >= 1:
                pt_arr = np.array(obj.pt)
                pt_arr[dr >= dRcut] = -1
                if np.max(pt_arr) > res12[np.argmax(pt_arr)]:
                    res12[np.argmax(pt_arr)] = idx
                res21[idx] = np.argmax(pt_arr)
        for elem in res12:
            if elem == -1:
                builder12.append(None)
            else:
                builder12.append(elem)
        for elem in res21:
            if elem == -1:
                builder21.append(None)
            else:
                builder21.append(elem)
        builder12.end_list()
        builder21.end_list()
    return builder12, builder21


# %%
GenEle = events.GenEl
GenEle = GenEle[np.abs(GenEle.eta) < 1.48]
inAcceptanceMask = ak.num(GenEle) > 0

GenEle = GenEle[inAcceptanceMask]
CryClu = events.CaloCryCluGCT[inAcceptanceMask]
Tk = events.DecTkBarrel[inAcceptanceMask]
TkEle = events.TkEleL2[inAcceptanceMask]
GenEle["phi"] = GenEle.calophi
GenEle["eta"] = GenEle.caloeta
Tk["phi"] = Tk.caloPhi
Tk["eta"] = Tk.caloEta


Tk = Tk.compute()
CryClu = CryClu.compute()
GenEle = GenEle.compute()
TkEle=TkEle.compute()



# %%


CryClu["GenIdx"], GenEle["CryCluIdx"] = label_builder(
    ak.ArrayBuilder(), ak.ArrayBuilder(), CryClu, GenEle, dRcut=0.2
)

Tk["CryCluIdx"], CryClu["TkIdx"] = label_builder(
    ak.ArrayBuilder(), ak.ArrayBuilder(), Tk, CryClu, dRcut=0.2
)
Tk["GenIdx"], GenEle["TkIdx"] = label_builder(
    ak.ArrayBuilder(), ak.ArrayBuilder(), Tk, GenEle, dRcut=0.2
)

TkEle["GenIdx"], GenEle["TkEleIdx"] = label_builder(
    ak.ArrayBuilder(), ak.ArrayBuilder(), TkEle, GenEle, dRcut=0.2
)


# %%


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


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


@nb.jit
def delta_phi(phi1, phi2):
    return (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi


@nb.jit
def delta_r(phi1, phi2, eta1, eta2):
    return np.sqrt((eta1 - eta2) ** 2 + delta_phi(phi1, phi2) ** 2)

#-1 = not matched, else index of the genEle
@nb.jit
def label_builder(builder, Obj, GenEle,dRcut):
    for event_idx in range(len(Obj)):
        builder.begin_list()
        obj = Obj[event_idx]
        res = np.ones(len(obj), dtype=nb.int32)*-1
        for idx,ele in enumerate(GenEle[event_idx]):
            dr = delta_r(
                np.array(ele.phi),
                np.array(obj.phi),
                np.array(ele.eta),
                np.array(obj.eta),
            )
            # For each genEle, match the objwith dR<0.2 and the hightest pt
            if np.sum(dr < dRcut) >= 1:
                pt_arr = np.array(obj.pt)
                pt_arr[dr >= dRcut] = -1
                res[np.argmax(pt_arr)] = idx
        for elem in res:
            builder.append(elem)
        builder.end_list()
    return builder


GenEle = events.GenEl.compute()
GenEle = GenEle[np.abs(GenEle.eta) < 1.48]
inAcceptanceMask = ak.num(GenEle) > 0

GenEle = GenEle[inAcceptanceMask]
CryClu = events.CaloCryCluGCT.compute()[inAcceptanceMask]
Tk = events.DecTkBarrel.compute()[inAcceptanceMask]
GenEle['phi']=GenEle.calophi
GenEle['eta']=GenEle.caloeta
Tk['phi']=Tk.caloPhi
Tk['eta']=Tk.caloEta

#%%
CryClu["GenIdx"] = label_builder(
    ak.ArrayBuilder(),
    CryClu,
    GenEle,
    dRcut=0.2
).snapshot()
Tk["CryCluIdx"] = label_builder(
    ak.ArrayBuilder(),
    Tk,
    CryClu,
    dRcut=0.4
).snapshot()
Tk["GenIdx"] = label_builder(
    ak.ArrayBuilder(),
    Tk,
    GenEle,
    dRcut=0.2
).snapshot()

#%%

def plot_efficiencies(Gen,CryClu,Tk,bins=50):
    tk_gen=GenEle[Tk.GenIdx[Tk.GenIdx > -1]]
    cc_gen=GenEle[CryClu.GenIdx[CryClu.GenIdx > -1]]
    tk_cc=CryClu[Tk.CryCluIdx[Tk.CryCluIdx > -1]]
    tk_cc_gen = GenEle[tk_cc.GenIdx[tk_cc.GenIdx > -1]]
    
    genHist, genEdges = np.histogram(Gen.pt, bins=bins, range=(0, 100))
    centers = (genEdges[:-1] + genEdges[1:]) / 2
    
    tk_genHist, _ = np.histogram(ak.flatten(tk_gen).pt, bins=genEdges)
    cc_genHist, _ = np.histogram(ak.flatten(cc_gen).pt, bins=genEdges)
    tk_cc_genHist, _ = np.histogram(ak.flatten(tk_cc_gen).pt, bins=genEdges)

    
    fig,ax=plt.subplots()
    ax.step(centers,cc_genHist/genHist,where="mid",label="CryClu-Gen")
    ax.step(centers,tk_genHist/genHist,where="mid",label="Tk-Gen")
    ax.step(centers,tk_cc_genHist/genHist,where="mid",label="Tk-CryClu-Gen")
    ax.legend()
    ax.set_xlabel("GenEle pt [GeV]")
    ax.set_ylabel("Efficiency")
    ax.grid()
plot_efficiencies(GenEle,CryClu,Tk,bins=100)
    
#probabilmente tk-cryclu-gen è sbagliato
# quello che faccio è per ogni cryclu, matcho un tk. Devovedificare che entrambi matchano lo stesso genEle

#%%
# %%
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import numba as nb


hep.style.use("CMS")
NanoAODSchema.warn_missing_crossrefs = False

dRcut = 0.2


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


@nb.jit
def label_builder(builder, Obj, GenEle):
    for event_idx in range(len(Obj)):
        builder.begin_list()
        obj = Obj[event_idx]
        res = np.zeros(len(obj), dtype=nb.int32)
        for ele in GenEle[event_idx]:
            dr = delta_r(
                np.array(ele.calophi),
                np.array(obj.phi),
                np.array(ele.caloeta),
                np.array(obj.eta),
            )
            # For each genEle, match the objwith dR<0.2 and the hightest pt
            if np.sum(dr < dRcut) >= 1:
                pt_arr = np.array(obj.pt)
                pt_arr[dr >= dRcut] = -1
                res[np.argmax(pt_arr)] = 1
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
# Tk["phi"] = Tk.caloPhi
# Tk["eta"] = Tk.caloEta

CryClu["label"] = label_builder(ak.ArrayBuilder(), CryClu, GenEle).snapshot()
Tk["label"] = label_builder(
    ak.ArrayBuilder(),
    Tk,
    GenEle,
).snapshot()

CryClu = CryClu[ak.argsort(CryClu.pt, ascending=False)]
Tk = Tk[ak.argsort(Tk.pt, ascending=False)]


# %%
#!plot for one event the Cry clu and the Tk in eta phi
def CryTk_etaphi(Gen, CryClu, Tk, nev, markersize=10):
    fig, ax = plt.subplots()
    ax.scatter(
        Gen[nev].caloeta,
        Gen[nev].calophi,
        s=markersize * Gen[nev].pt,
        label="GenEle",
        color="black",
    )
    scatter = ax.scatter(
        CryClu[nev].eta,
        CryClu[nev].phi,
        s=markersize * CryClu[nev].pt,
        label="CryClu",
        color="dodgerblue",
        alpha=0.7,
    )
    ax.scatter(
        Tk[nev].eta,
        Tk[nev].phi,
        s=markersize * Tk[nev].pt,
        label="Tk",
        color="red",
        alpha=0.7,
    )

    for gen in Gen[nev]:
        circle = plt.Circle(
            (gen.caloeta, gen.calophi), dRcut, color="black", fill=False
        )
        ax.add_artist(circle)
    legend1 = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.025),
        fancybox=True,
        shadow=True,
        ncol=5,
    )
    ax.set_xlabel("$\eta$")
    ax.set_ylabel("$\phi$")
    plt.grid()
    hep.cms.text("Phase2 Simulation", ax=ax)
    hep.cms.lumitext(f"PU=200, $\Delta R={dRcut}$", ax=ax)
    ax.set_axisbelow(True)

    handles, labels = scatter.legend_elements(prop="sizes")
    ax.legend(
        handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), title="$p_T$ [GeV]"
    )
    ax.add_artist(legend1)
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-3.14, 3.14)


def dR_dPt(Gen, CryClu, Tk, nev):
    cc = CryClu[nev]
    tk = Tk[nev]

    CryCluSig = cc[cc.label == 1]
    
    fig,ax =plt.subplots(2,1)
    for idx,cryclu in enumerate(CryCluSig):
        dr=delta_r(np.array(cryclu.eta),
                    np.array(tk.eta),
                    np.array(cryclu.phi),
                    np.array(tk.phi)
                    )
        drSig = dr[tk.label == 1]
        drBkg = dr[tk.label == 0]
        
        dpt= cryclu.pt-tk.pt
        dptSig = dpt[tk.label == 1]
        dptBkg = dpt[tk.label == 0]

        ax[idx].scatter(
            drSig,
            dptSig,
            color="red",
            alpha=1,
            s=80
        )
        ax[idx].scatter(
            drBkg,
            dptBkg,
            color="dodgerblue",
            alpha=0.5,
            s=40
        )
        ax[idx].grid()
        ax[idx].set_xlabel("$\Delta R$")
        ax[idx].set_ylabel("$\Delta p_T [GeV]$")
        ax[idx].set_xlim(0,6)
        ax[idx].set_ylim(-80,80)
        #ax[idx].set_xscale("log")


for i in range(10):
    CryTk_etaphi(GenEle, CryClu, Tk, i)
    dR_dPt(GenEle, CryClu, Tk, i)
    plt.show()



#%%

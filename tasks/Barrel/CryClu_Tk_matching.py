
#%%
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
            if np.sum(dr < dRcut) >= 1:
                pt_arr = np.array(obj.pt)
                pt_arr[dr >= dRcut] = -1
                res[np.argmax(pt_arr)] = 1
        for elem in res:
            builder.append(elem)
        builder.end_list()
    return builder


CryClu = events.CaloCryCluGCT.compute()
Tk = events.DecTkBarrel.compute()
Tk["phi"] = Tk.caloPhi
Tk["eta"] = Tk.caloEta
GenEle = events.GenEl.compute()

CryClu_label=label_builder(ak.ArrayBuilder(),
                           CryClu,
                           GenEle).snapshot()
Tk_label = label_builder(ak.ArrayBuilder(),
                         Tk,
                         GenEle,).snapshot()




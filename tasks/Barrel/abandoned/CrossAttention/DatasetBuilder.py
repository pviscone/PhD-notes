#%%
import numpy as np
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numba as nb
import pandas as pd
import pickle

events = NanoEventsFactory.from_root(
    {"Nano.root": "Events"},
    schemaclass=NanoAODSchema,
).events()

CryClu=events.CaloCryCluGCT.compute()
GenEle=events.GenEl.compute()
Tk=events.DecTkBarrel.compute()

def toCaloVar(arr):
    arr["eta"]=arr.caloeta
    arr["phi"]=arr.calophi
    return arr

GenEle=toCaloVar(ak.with_name(GenEle,"PtEtaPhiMLorentzVector"))
CryClu=ak.with_name(CryClu,"PtEtaPhiMLorentzVector")
GenEle=GenEle[np.abs(GenEle.eta)<1.49]


#%%
dRcut=0.2

@nb.jit
def delta_phi(phi1,phi2):
    return (phi1-phi2+np.pi)%(2*np.pi)-np.pi

@nb.jit
def delta_r(phi1,phi2,eta1,eta2):
    return np.sqrt((eta1-eta2)**2+delta_phi(phi1,phi2)**2)

@nb.jit
def label_builder(builder, CryClu,GenEle):
    for event_idx in (range(len(CryClu))):
        builder.begin_list()
        cryclu=CryClu[event_idx]
        res=np.zeros(len(cryclu),dtype=nb.int32)
        for ele in GenEle[event_idx]:
            dr=delta_r(np.array(ele.phi),
                       np.array(cryclu.phi),
                        np.array(ele.eta),
                        np.array(cryclu.eta))
            if np.sum(dr<dRcut)>=1:
                pt_arr=np.array(cryclu.pt)
                pt_arr[dr>=dRcut]=-1
                res[np.argmax(pt_arr)]=1
        for elem in res:
            builder.append(elem)
        builder.end_list()
    return builder

@nb.jit
def genpt_builder(builder, CryClu,GenEle):
    for event_idx in (range(len(CryClu))):
        builder.begin_list()
        cryclu=CryClu[event_idx]
        res=np.zeros(len(cryclu),dtype=nb.int32)
        for ele in GenEle[event_idx]:
            dr=delta_r(np.array(ele.phi),
                       np.array(cryclu.phi),
                        np.array(ele.eta),
                        np.array(cryclu.eta))
            if np.sum(dr<dRcut)>=1:
                pt_arr=np.array(cryclu.pt)
                pt_arr[dr>=dRcut]=-1
                res[np.argmax(pt_arr)]=ele.pt
        for elem in res:
            builder.append(elem)
        builder.end_list()
    return builder




labels=label_builder(ak.ArrayBuilder(), CryClu, GenEle).snapshot()
CryClu['labels']=labels

#%%


#!mvaQual is always 0
#!electronWP is always 0
#!photonWP is always 0
#!stage2effMatch is always 0
#!.looseL1TkMatchWP is always 0
#! hovere is always 0
#! pucorrpt is always 0
#! bremstrength is always 0

varDict={
    "Tk":["hitPattern",
          "pt",
          "eta",
          "phi",
          "caloEta",
          "caloPhi",
          "nStubs",
          "chi2Bend",
          "chi2RPhi",
          "chi2RZ",
          "vz"],
    "CryClu":["standaloneWP",
            'calibratedPt',
            'e2x5',
            'e5x5',
            'eta',
            'isolation',
            'phi',
            'pt',
            'labels'
              ],
}


n_events=len(CryClu)
dfDict={}
for obj in varDict:
    dt=[(var,'f4') for var in varDict[obj]]
    exec(f'nmax=ak.max(ak.num({obj}))')
    exec(f'ak_obj=ak.pad_none({obj},nmax)')
    exec(f'np_arr=np.zeros((n_events,len(varDict["{obj}"]),nmax))')
    for idx, var in enumerate(varDict[obj]):
        print(var)
        exec(f'np_arr[:,idx]=ak.to_numpy(ak_obj[var]).filled(-999)')
    dfDict[obj]=np.swapaxes(np_arr,1,2)


# %%
toSaveDict={"data":dfDict,
            "columns":varDict,
            'genpt':genpt_builder(ak.ArrayBuilder(), CryClu, GenEle).snapshot()}
pickle.dump(toSaveDict, open('pickle_data.p', 'wb'))


#%%



#%%

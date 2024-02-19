#%%
import numpy as np
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam

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

CryCluNear,dR=CryClu.nearest(GenEle,return_metric=True)
labels=(dR<dRcut)

#!mvaQual is always 0
#!electronWP is always 0
#!photonWP is always 0
#!stage2effMatch is always 0
#!.looseL1TkMatchWP is always 0
#! hovere is always 0
varDict={
    "Tk":["hitPattern",
          "pt",
          "eta",
          "phi",
          "caloEta",
          "caloPhi",
          "hitPattern",
          "nStubs",
          "chi2Bend",
          "chi2RPhi",
          "chi2RZ",
          "vz"],
    "CryClu":["standaloneWP",
              'bremStrength',
            'calibratedPt',
            'e2x5',
            'e5x5',
            'eta',
            'isolation',
            'phi',
            'pt',
            'puCorrPt'
              ],
    'labels':labels
}
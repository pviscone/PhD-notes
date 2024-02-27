#%%
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


hep.style.use("CMS")
NanoAODSchema.warn_missing_crossrefs = False


events = NanoEventsFactory.from_root(
    {"Nano.root": "Events"},
    schemaclass=NanoAODSchema,
).events()


GenEle=events.GenEl.compute()
CryCluGCT=events.CaloCryCluGCT.compute()

def toCaloVar(arr):
    arr["eta"]=arr.caloeta
    arr["phi"]=arr.calophi
    return arr

GenEle=toCaloVar(ak.with_name(GenEle,"PtEtaPhiMLorentzVector"))
CryCluGCT=ak.with_name(CryCluGCT,"PtEtaPhiMLorentzVector")


GenEle=GenEle[np.abs(GenEle.eta)<1.49]


#%%
dRcut=0.2
CryCluGCTNear,dRGCT=GenEle.nearest(CryCluGCT,return_metric=True)


GenEle=ak.flatten(GenEle,axis=1)
dRGCT=ak.flatten(dRGCT,axis=1)


#%%
bins=50
histrange=(1,100)
genHist=plt.hist(GenEle.pt,histtype="step",bins=bins,range=histrange,label="GenEle",linewidth=1.5)

edges=genHist[1]
centers=(edges[:-1]+edges[1:])/2
gen=genHist[0]

#!NON dovrebbero essere flat in pt? le fluttuazioni sono molto maggiori delle barre di errore
plt.errorbar(centers,gen,yerr=gen**0.5,fmt="none",label=None)

gct=plt.hist(GenEle[dRGCT<dRcut].pt,histtype="step",bins=bins,range=histrange,label="CryCluGCTV2",linewidth=1.5)[0]
plt.grid()
plt.legend()
plt.xlabel("GenEle_Pt [GeV]")
hep.cms.text("Phase2 Simulation")
hep.cms.lumitext(f"$\Delta R_{{CC-GEN}}<${dRcut}")

#%%
plt.step(centers,gct/gen,label="CryCluGCT",where="mid")

hep.cms.text("Phase2 Simulation")
hep.cms.lumitext(f"$\Delta R_{{CC-GEN}}<${dRcut}")
plt.xlabel("GenEle_Pt [GeV]")
plt.ylabel("Efficiency")
plt.legend()
plt.grid()
plt.show()

#%%
bins=25
plt.figure()
plt.subplot(1,2,1)
a=plt.hist2d(ak.to_numpy(GenEle.phi),ak.to_numpy(GenEle.eta),bins=bins,range=((-3.15,3.15),(-1.5,1.5)),cmap="viridis",label="GenEle")
plt.subplot(1,2,2)
b=plt.hist2d(ak.to_numpy(GenEle[dRGCT<dRcut].phi),ak.to_numpy(GenEle[dRGCT<dRcut].eta),bins=bins,range=((-3.15,3.15),(-1.5,1.5)),cmap="viridis",label="GenEle")

plt.figure(figsize=(13,5))
c=b[0]/a[0]
plt.imshow(c,cmap="viridis",extent=(-3.15,3.15,-1.5,1.5),origin="lower")
plt.xlabel("$\phi$")
plt.ylabel("$\eta$")
plt.colorbar()

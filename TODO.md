### ASAP

- [x] Scrivere mail Jin Li (purezza + tag sbagliato)
  
  - [ ] Reply to the email

- [x] Syncare FastPuppi

- [x] Generare ntuple con submission e gli inputfile di gianluca

### NanoHistDump

- Dask
  
  - [ ] Capire cosa sia delayed
  
  - [ ] Vedere come gestire npartition e in generale il grafo, capire se é feasible

- In alternativa
  
  - [ ] Piallare dask e fare compute in lettura
  
  - [ ] Leggere solo le collection presenti nel dict schema

- Implementare
  
  - [ ] Loop su selezioni

- [ ] Reimplement renaming
  
  > Hi all! I hope you're having a wonderful day :slightly_smiling_face:
  > 
  > Is it possible/what are the ways to rename a dask awkward array read by Coffea? Perhaps Coffea has some mechanism to do this (while reading a tree)?
  > 
  > 1. We are reading it via from_root
  >    
  >    `akarray_new = NanoEventsFactory.from_root(self.tree, schemaclass=BaseSchema).events()`
  > 
  > self.tree is a **uproot.models.TTree.Model_TTree_v20 .** It is a tree from a .ROOT file.
  > 
  > 1. Before that we used:
  > 
  > `akarray_old = self.tree.arrays(names, library='ak', aliases=name_map, entry_start=self.file_entry, entry_stop=self.file_entry+entry_block)`
  > 
  > Where the aliases parameter would do the renaming for us. The structure being:
  > 
  > New field / old field (currently in the dask awkward array)
  > 
  > > {'eta': 'TkEleL2_eta',
  > >  'pfIso': 'TkEleL2_pfIso',
  > >  'pfIsoPV': 'TkEleL2_pfIsoPV',
  > >  'phi': 'TkEleL2_phi',
  > >  'pt': 'TkEleL2_pt',
  > >  'puppiIso': 'TkEleL2_puppiIso',
  > >  'puppiIsoPV': 'TkEleL2_puppiIsoPV',
  > >  'tkEta': 'TkEleL2_tkEta',
  > >  'tkIso': 'TkEleL2_tkIso',
  > >  'tkIsoPV': 'TkEleL2_tkIsoPV',
  > >  'tkPhi': 'TkEleL2_tkPhi',
  > >  'tkPt': 'TkEleL2_tkPt',
  > >  'vz': 'TkEleL2_vz',
  > >  'charge': 'TkEleL2_charge',
  > >  'hwQual': 'TkEleL2_hwQual'}.
  > 
  > Not doing the renaming would be very hard, as the generic .pt, .eta, .charge and others fields are used in many many different parts of the code.
  > 
  > P.S: I have a jupyter notebook prepared (in the dms), for who are interested.
  > 
  > All information is welcome. Cheers. Edited
  > 
  > Show more
  > 
  > Lindsey Gray
  > 
  > [14:54](https://mattermost.web.cern.ch/cms-exp/pl/4tn31x5d67fk8jdnx53x91qh1y)
  > 
  > With the naming scheme in the file you should be able to use NanoAODSchema. You'll have to add `TkEleL2` to NanoAODSchema.mixins ([coffea/src/coffea/nanoevents/schemas/nanoaod.py at master · CoffeaTeam/coffea · GitHub](https://github.com/CoffeaTeam/coffea/blob/master/src/coffea/nanoevents/schemas/nanoaod.py#L53)). Just set that before you make the NanoEvents instance. You can probably safely use PtEtaPhiMCollection as the target type for branches with various prefixes you're looking for.
  > 
  > [14:54](https://mattermost.web.cern.ch/cms-exp/pl/hhdowkeg4pnujdcdqjya1nt9ha)
  > 
  > There does need to be a `nTkEleL2` counter branch though. Does your tree have that?

### Plots

- [ ] Fare classi per plotting

- [ ] Capire come fare versioning plot

### Real Work

- [ ] Plot molteplicitá di tutti gli oggetti in funzione di genpt

- [ ] Plot molteplicitá di tutti gli oggetti in funzione di dpt

- [ ] Plot risoluzione eta/phi vs pt rispetto ai gen per capire come reimplementare elliptic matching in funzione di pt

- [ ] Fai il check che la scelta del matched avvenga tramite min dpt

- [ ] Studio purezza (dopo mail jin li)

## Idee analisi

- tauonio ee o gamma gamma (prodotto a soglia leptoni soft) (David d'enterria)

- rare meson decay (maybe not CMG)

- bph rk with parking dataset (Paris)

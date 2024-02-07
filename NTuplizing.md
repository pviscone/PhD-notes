Script responsabile della Ntuplizzazione: Runperformancentuple.py in fast puppi

Task:
- Aggiungere funzione customize per aggiungere flat table (prendi ispirazione da addtkem per salvare i crystal cluster)
- Aggiungere alle flag  la funzione add crystal cluster create (vedi script HTCondor di gianluca)
- Sarebbe bene aggiungere pure questi allo skim e alle ntuple 

```
keep *_l1tPhase2L1CaloEGammaEmulator_*_*
```


- Fai plot di efficenza in gen pt dei tkelectroneb nel barrel, i crystal cluster e le tracce con eta<1.4 


[https://github.com/cms-sw/cmssw/tree/master/L1Trigger/Phase2L1ParticleFlow/python](https://github.com/cms-sw/cmssw/tree/master/L1Trigger/Phase2L1ParticleFlow/python)

Oggetti correlator:
[https://github.com/cms-sw/cmssw/tree/master/DataFormats/L1TParticleFlow/interface](https://github.com/cms-sw/cmssw/tree/master/DataFormats/L1TParticleFlow/interface)

Correlator e/g objects: [https://github.com/cms-sw/cmssw/tree/master/DataFormats/L1TCorrelator/interface](https://github.com/cms-sw/cmssw/tree/master/DataFormats/L1TCorrelator/interface)

[https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1CaloTrigger/python/l1tEGammaCrystalsEmulatorProducer_cfi.py](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1CaloTrigger/python/l1tEGammaCrystalsEmulatorProducer_cfi.py)

[https://github.com/cms-sw/cmssw/blob/master/DataFormats/L1TCalorimeterPhase2/interface/CaloCrystalCluster.h](https://github.com/cms-sw/cmssw/blob/master/DataFormats/L1TCalorimeterPhase2/interface/CaloCrystalCluster.h)

Aggiungere customize per flat table che salvi tutto il contenuto dei CrystalClusters nelle ntuple di fastpuppi

[https://github.com/p2l1pfp/FastPUPPI/blob/14_0_X/NtupleProducer/python/runPerformanceNTuple.py#L500](https://github.com/p2l1pfp/FastPUPPI/blob/14_0_X/NtupleProducer/python/runPerformanceNTuple.py#L500)


 c’e’ una nuova versione dei XstalClusters prodotta dal producer:

[https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1CaloTrigger/plugins/Phase2L1CaloEGammaEmulator.cc](https://github.com/cms-sw/cmssw/blob/master/L1Trigger/L1CaloTrigger/plugins/Phase2L1CaloEGammaEmulator.cc)
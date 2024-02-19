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


## File
/eos/cms/store/cmst3/group/l1tr/cerminar/14_0_X/fpinputs_131X/v2/DoubleElectron_FlatPt-1To100_PU200/inputs131X_10.root

NON FUNZIONA, usa quelli di default presenti nello script
## Errori compilazione fastpuppi

Se hai degli errori su vocms1000 compila tramite singularity cmssw-el8 -- scram b -j 4

-lcrypto
-lssl


```
[Type/Paste Your Code](<%3E> Building shared library tmp/el8_amd64_gcc12/src/FastPUPPI/NtupleProducer/src/FastPUPPINtupleProducer/libFastPUPPINtupleProducer.so
/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/gcc/12.3.1-40d504be6370b5a30e3947a6e575ca28/bin/../lib/gcc/x86_64-redhat-linux-gnu/12.3.1/../../../../x86_64-redhat-linux-gnu/bin/ld.bfd: cannot find -lssl: No such file or directory
/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/gcc/12.3.1-40d504be6370b5a30e3947a6e575ca28/bin/../lib/gcc/x86_64-redhat-linux-gnu/12.3.1/../../../../x86_64-redhat-linux-gnu/bin/ld.bfd: cannot find -lcrypto: No such file or directory
collect2: error: ld returned 1 exit status
gmake: *** [config/SCRAM/GMake/Makefile.rules:1803: tmp/el8_amd64_gcc12/src/FastPUPPI/NtupleProducer/src/FastPUPPINtupleProducer/libFastPUPPINtupleProducer.so] Error 1
gmake: *** [There are compilation/build errors. Please see the detail log above.] Error 2>)
```


---
Le funzioni bisogna chiamarle alla fine dello script. Non conviene fare un accrocchio con argparse?


---

Aggiunto"keep *_l1tPhase2L1CaloEGammaEmulator_*_*", allo skim

CaloCrystalCluster dataformat:
https://github.com/cms-sw/cmssw/blob/93f14ae8b3fa6fb1cf1a14c73f651f6b01d17340/DataFormats/L1TCalorimeterPhase2/interface/CaloCrystalCluster.h#L100

Phase2L1CaloEGammaEmulator.cc producer source 
(
da qui prendo label GCT linea 633 (guarda anche 502)
Constructor definition linea 455)
https://github.com/cms-sw/cmssw/blob/93f14ae8b3fa6fb1cf1a14c73f651f6b01d17340/L1Trigger/L1CaloTrigger/plugins/Phase2L1CaloEGammaEmulator.cc#L95


EDProducer in .py di Phase2L1CaloEGammaEmulator non pervenuto. Esiste l1tPhase2L1CaloEGammaEmulator che pero prende da Phase2L1CaloPFClusterEmulator


Pull request che ha aggiunto i cosicluster
https://github.com/cms-l1t-offline/cmssw/pull/1069/files

Esempio di analyzer sui calocluster
https://github.com/skkwan/phase2-l1Calo-analyzer
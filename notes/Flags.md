Flag:

- looseL1TkMatchWP
- electronWP90/98
- photonWP80
- stage2effMatch
- standaloneWP

Producers
- L1EGCrystalClusterEmulatorProducer
- Phase2L1CaloEGammaEmulator
    

# [L1EGCrystalClusterEmulatorProducer](https://github.com/cms-sw/cmssw/blob/ca93802f7a4edafab85b72895b4b877df8255d0e/L1Trigger/L1CaloTrigger/plugins/L1EGammaCrystalsEmulatorProducer.cc#L264)

### [looseL1TkMatchWP](https://github.com/cms-sw/cmssw/blob/ca93802f7a4edafab85b72895b4b877df8255d0e/L1Trigger/L1CaloTrigger/plugins/L1EGammaCrystalsEmulatorProducer.cc#L1146):
```cpp
     int looseL1TkMatchWP = (int)(is_looseTkiso && is_looseTkss);
```
- [is_looseTkiso](https://github.com/cms-sw/cmssw/blob/ca93802f7a4edafab85b72895b4b877df8255d0e/L1Trigger/L1CaloTrigger/plugins/L1EGammaCrystalsEmulatorProducer.cc#L1111):

```cpp
		bool is_looseTkiso =passes_looseTkiso(energy_cluster_L2Card[ii][jj][ll], 
									  isolation_cluster_L2Card[ii][jj][ll]);
```

	-  



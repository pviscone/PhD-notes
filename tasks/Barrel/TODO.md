Idee dalla piú facile alla piú complicata

1. Semplice matching con dR dicryclu e tracce passando dR e dpt tra traccia e cluster alla rete.
Capire come gestire mancanza di traccia matchata
Il matching va fatto al cluster, forse per quello usare dr=0.4 (NO!!!!, prendi tracce che non c'entrano nulla)

2. Matching tramite 1-nn nel piano dr-dpt,riscalando gli assi in modo da ottimizzare il matching.Passando sempre dpt e dr alla rete

3. Knn e via di transformer o GNN

4. Effettuare un reclustering 


TODO:
[] Studiare bene efficienze matching:
    - CryClu al Gen
    - Tk al CryClu
    - Tk al CryClu con verifica del matching al gen
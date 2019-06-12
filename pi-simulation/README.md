# Simulation de pi sur GPU

## Description : 
simple méthode de monte-carlo qui utilise cuda pour approximer pi. Utiliser "make pi-simulation" pour compiler le fichier.

Temps de calcul de la tâche sur GPU lors de la méthode par nombre de simulations :
Pour 1 000 000   : 1.95e-05
Pour 10 000 000  : 2.54e-05
Pour 100 000 000 : 2.35e-05


## Questions :
- ligne 15 : A quoi sert cette ligne ?
- ligne 66 : Comment calculer le threadsperBlock idéal ? 
- ligne 67 : Pourquoi calcule-t-on le blocksperGrid comme ça ?
- Comment interpréter le temps de calcul ?

# Test de programmes simples utilisant mshadow
utiliser "make all" pour compiler tous les programmes. 

## Remarques sur les programmes
- **simple_allocator** : montre que quand on initialise un tenseur avec un tableau de données, il faut le resizer correctement ensuite. Bonne méthode ? A voir s'il n'y a pas plus simple. *dptr_* : pointeur sur les données, et ensuite on reshape comme on veut pour faire les bonnes opérations. 

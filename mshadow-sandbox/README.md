# Test de programmes simples utilisant mshadow
utiliser "make all" pour compiler tous les programmes. 

## Remarques sur les programmes
- **simple_allocator** : montre que quand on initialise un tenseur avec un tableau de données, il faut le resizer correctement ensuite. Pourquoi ? Il semblerait que les objets du type Tensor<gpu, 2, float> soient des sortes de pointeurs, et qu'ils ne sont là que pour donner une indication au programme sur la façon de gérer les opérations. Donc les champs important d'un Tensor sont :
  - dptr_ : pointe vers le tableau de données désiré
  - shape_ : la dimension du tenseur
  - stride_ : ???

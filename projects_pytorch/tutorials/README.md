# Description des programmes

- **tensor_example.cpp** :
  - Utilisation d'opérations usuelles sur les tenseurs. La plupart des opérateurs sont surchargés de manière *elementwise*.
  - Utilisation de l'allocateur host/device par *torch::Device*. Question : comment utiliser l'allocateur sur GPU à bon escient ?
  - Exemple de *forward propagation* simple sur un tenseur random. L'application de fonctions d'activation est particulièrement simple.
  - L'objet principal de *libtorch* est le **Module**. Il est composé de sous-modules, et de paramètres qui simplifieraient le calcul d'un gradient. Question : creuser la structure du module et comprendre son fonctionnement.
- **autograd_example.cpp** : Utilisation de la méthode *backward()* pour calculer le gradient d'une expression et comparaison avec un calcul de gradient classique.
  - Méthode : initialiser le tenseur avec l'option *requires_grad(true)*, utiliser la méthode *backward()* et faire un appel au gradient de la variable par la méthode *grad()*.
  - La méthode backward est **récursive** et c'est tout l'intérêt d'*autograd* : si on calcule f1(x) puis f2(x) = g(f1(x)), la méthode backward() donnera le gradient pour x sans problème puisque les informations sur df1/dx sont stockées dans l'objet f1 lors de sa création. 
  - Question : il y a un léger écart entre la méthode de calcul de gradient normal et backward lorsqu'on a des expressions plus longues. Ceci est clairement dû au *Dtype* mais quelle est la meilleure approximation ?
- **GPU_CPU_example.cpp** : Estimation de pi par CPU puis par GPU pour comparer les méthodes et temps de calcul.
  -  Syntaxe pour faire passer un tenseur sur GPU très simple. Pour cet exercice, le temps de calcul le plus 'rapide' est donné quand on met les premiers tenseurs (les random) sur GPU sans faire rien d'autre. Meilleure pratique ? A creuser.
  - Question : impossible de faire une estimation de plus de 200,000,000, même sur CPU. Limitations du tenseur ? A creuser.
- **nnet_from_scratch.cpp** : *en construction*

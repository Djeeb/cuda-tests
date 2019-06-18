# Description des programmes

- **tensor_example.cpp** :
  - Utilisation d'opérations usuelles sur les tenseurs. La plupart des opérateurs sont surchargés de manière *elementwise*.
  - Utilisation de l'allocateur host/device par *torch::Device*. Question : comment utiliser l'allocateur sur GPU à bon escient ?
  - Exemple de *forward propagation* simple sur un tenseur random. L'application de fonctions d'activation est particulièrement simple.
  - L'objet principal de *libtorch* est le **Module**. Il est composé de sous-modules, et de paramètres qui simplifieraient le calcul d'un gradient. Question : creuser la structure du module et comprendre son fonctionnement.
 **autograd_example.cpp** : *en construction*
  **test_GPU.cpp** : *en construction*

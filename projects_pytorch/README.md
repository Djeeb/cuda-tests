# Remarques sur libtorch
- **But de libtorch** : c'est une API C++ de Pytorch créée en premier lieu pour des cas particuliers pouvant difficilement se traiter en python (utilisation de plusieurs GPU, besoin de haute performance, rapidité, etc.). Cela a deux conséquences :
  - La librairie est stable et très fournie (justement pour traiter les cas particuliers).
  - Il y a peu de documentation. Il ne faut donc pas hésiter à chercher soi-même sur internet ou à tester des programmes rapides pour comprendre le fonctionnement de telle classe ou fonction. 

# Description des projets

- **tutorials** : des programmes simples pour comprendre la syntaxe et le fonctionnement de libtorch. 
- **nnet_from_scratch** : implémentation d'un réseau de neurones à une couche n'utilisant que la structure des tenseurs. Permet de comprendre toutes les étapes d'une phase d'apprentissage profond. Entraînement sur MNIST. 

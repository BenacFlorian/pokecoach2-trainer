Ce projet servira a entrainer un nouveau model / finetuner l'ancien pour chaque nouvelle regulation. 

Dans quel ordre je construirais le Trainer ?
Je procéderais par paliers pour limiter les risques.

Premier palier : récupérer le code de VGC-Bench, reproduire son entraînement BC sur un petit dataset existant, inspecter le masquage et vérifier qu'un checkpoint produit est compatible avec PokeCoach.

Deuxième palier : développer le Trainer FastAPI, les jobs de collecte et de conversion, la gestion des expériences, la journalisation et le téléchargement des modèles.

Troisième palier : rendre l'architecture paramétrable, entraîner plusieurs tailles et mettre en place les comparaisons automatiques.

Quatrième palier : construire des datasets propres aux nouvelles régulations et automatiser leur mise à jour.

Le point sur lequel j'insisterais est le suivant : le Trainer doit être une application indépendante, mais il doit partager un contrat de modèle strictement compatible avec PokeCoach.

Chaque modèle publié devrait conserver son architecture, la version de l'encodage des observations, celle de l'espace d'actions, ses hyperparamètres, ses filtres de données et ses métriques d'évaluation.
# Fuite de données

Une fuite de données (data leakage) se produit quand une information du jeu de test, ou une information indisponible au moment de la prédiction, se retrouve dans l'entraînement.
Les scores deviennent excellents en test puis s'effondrent en production. Exemple classique : normaliser les données avant de séparer entraînement et test.

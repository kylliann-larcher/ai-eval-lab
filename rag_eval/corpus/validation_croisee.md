# Validation croisée

La validation croisée découpe les données en k blocs. On entraîne k fois le modèle en gardant à chaque fois un bloc différent pour l'évaluation, puis on fait la moyenne des scores.
Elle donne une estimation plus stable qu'un seul découpage, surtout sur un petit jeu de données. Sur des classes déséquilibrées, on utilise une version stratifiée pour garder la même proportion de chaque classe dans chaque bloc.

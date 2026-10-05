# Classes déséquilibrées

On parle de classes déséquilibrées quand une classe est beaucoup plus rare que l'autre, par exemple 1 anomalie pour 99 points normaux.
Les remèdes courants : pondérer les classes pendant l'entraînement (class_weight), rééchantillonner, ajuster le seuil de décision, et évaluer avec le rappel, la précision ou l'aire sous la courbe précision-rappel plutôt qu'avec l'accuracy.

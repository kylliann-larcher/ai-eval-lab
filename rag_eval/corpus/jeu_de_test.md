# Jeu de test de référence

Un jeu de test de référence (golden set) est une liste fixe de cas avec la réponse attendue, écrite et vérifiée à la main.
Pour une application LLM, 30 à 50 questions réelles suffisent pour commencer. On le relance à chaque changement de prompt, de modèle ou de données, et on compare au run précédent pour repérer les régressions.

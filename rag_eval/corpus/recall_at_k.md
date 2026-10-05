# Recall@k pour la recherche documentaire

Dans un système RAG, le recall@k vaut 1 si le bon document figure parmi les k premiers résultats du moteur de recherche, 0 sinon, moyenné sur toutes les questions.
Le MRR (mean reciprocal rank) tient compte du rang : 1 si le bon document arrive premier, 1/2 s'il arrive deuxième, et ainsi de suite.
Il faut mesurer la recherche séparément de la génération : si le bon document n'est jamais récupéré, aucun prompt ne sauvera la réponse.

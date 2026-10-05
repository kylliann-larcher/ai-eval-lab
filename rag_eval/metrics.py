"""Métriques d'évaluation pour un pipeline RAG (recherche + réponse)."""

from __future__ import annotations

import unicodedata


def recall_at_k(ranked_ids: list[str], relevant_id: str, k: int) -> float:
    """1.0 si le bon document est dans les k premiers résultats, 0.0 sinon."""
    return 1.0 if relevant_id in ranked_ids[:k] else 0.0


def reciprocal_rank(ranked_ids: list[str], relevant_id: str) -> float:
    """1/rang du bon document (1 s'il est premier, 1/2 s'il est deuxième...), 0 s'il est absent."""
    try:
        return 1.0 / (ranked_ids.index(relevant_id) + 1)
    except ValueError:
        return 0.0


def normalize(text: str) -> str:
    """Minuscules, sans accents, espaces insécables remplacés."""
    text = text.replace(" ", " ").replace(" ", " ").lower()
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def keywords_match(answer: str, keywords: list[str]) -> bool:
    """La réponse est jugée correcte si elle contient tous les mots-clés attendus."""
    norm = normalize(answer)
    return all(normalize(kw) in norm for kw in keywords)


def compare_runs(baseline: dict[str, bool], current: dict[str, bool]) -> dict[str, list[str]]:
    """Compare deux runs question par question.

    Renvoie les régressions (juste avant, faux maintenant) et les corrections
    (faux avant, juste maintenant). C'est ce qu'on regarde après chaque
    changement de prompt, de modèle ou de données.
    """
    common = sorted(set(baseline) & set(current))
    return {
        "regressions": [q for q in common if baseline[q] and not current[q]],
        "fixes": [q for q in common if not baseline[q] and current[q]],
    }

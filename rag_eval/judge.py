"""LLM comme juge, optionnel.

Un juge LLM note une réponse « correcte » ou « incorrecte » par rapport à la
réponse attendue. Ces juges ont des biais (réponses longues, ordre de
présentation...), donc on mesure d'abord leur accord avec des annotations
humaines avant de s'y fier : voir `agreement()`.

Le juge Anthropic n'est utilisé que si le paquet `anthropic` est installé et
que la variable d'environnement ANTHROPIC_API_KEY est définie.
"""

from __future__ import annotations

import os
from typing import Callable

from sklearn.metrics import cohen_kappa_score

JUDGE_PROMPT = """Tu évalues la réponse d'un assistant.

Question : {question}
Éléments attendus dans une bonne réponse : {expected}
Réponse de l'assistant : {answer}

La réponse contient-elle les éléments attendus, sans erreur factuelle ?
Réponds par un seul mot : CORRECT ou INCORRECT."""

Judge = Callable[[str, str, str], bool]


def anthropic_judge(model: str | None = None) -> Judge:
    """Crée un juge basé sur l'API Anthropic. Le modèle se règle avec JUDGE_MODEL."""
    import anthropic  # import local : dépendance optionnelle

    client = anthropic.Anthropic()
    model = model or os.environ.get("JUDGE_MODEL", "claude-haiku-4-5")

    def judge(question: str, expected: str, answer: str) -> bool:
        msg = client.messages.create(
            model=model,
            max_tokens=5,
            messages=[{
                "role": "user",
                "content": JUDGE_PROMPT.format(question=question, expected=expected, answer=answer),
            }],
        )
        return msg.content[0].text.strip().upper().startswith("CORRECT")

    return judge


def agreement(judge_labels: list[bool], human_labels: list[bool]) -> dict[str, float]:
    """Accord entre le juge et les humains : taux d'accord brut et kappa de Cohen.

    Le kappa corrige l'accord dû au hasard : 1 = accord parfait, 0 = pas mieux
    que le hasard. Si toutes les étiquettes sont identiques, le kappa n'est pas
    défini et vaut NaN.
    """
    if len(judge_labels) != len(human_labels) or not judge_labels:
        raise ValueError("Il faut deux listes non vides de même longueur.")
    raw = sum(j == h for j, h in zip(judge_labels, human_labels)) / len(judge_labels)
    if len(set(judge_labels) | set(human_labels)) < 2:
        kappa = float("nan")
    else:
        kappa = float(cohen_kappa_score(human_labels, judge_labels))
    return {"raw_agreement": raw, "cohen_kappa": kappa}

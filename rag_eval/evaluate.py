"""Évaluation minimale d'un pipeline RAG sur un jeu de test de référence.

On mesure séparément :
  - la recherche : le bon document est-il retrouvé ? (recall@1, recall@3, MRR)
  - la réponse : contient-elle les éléments attendus ? (mots-clés, ou juge LLM)

Chaque run est sauvegardé en JSON. Avec --baseline, on compare au run
précédent question par question pour repérer les régressions.

Usage :
    python -m rag_eval.evaluate --retriever words
    python -m rag_eval.evaluate --retriever chars --baseline results/rag_words.json
    python -m rag_eval.evaluate --retriever chars --labels rag_eval/labels_chars.json
    python -m rag_eval.evaluate --retriever chars --judge   # nécessite ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_eval.metrics import compare_runs, keywords_match, recall_at_k, reciprocal_rank

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "corpus"
GOLDEN_SET = ROOT / "golden_set.jsonl"
RESULTS_DIR = ROOT.parent / "results"

RETRIEVERS = {
    # TF-IDF sur les mots : rapide, mais rate les reformulations
    "words": dict(analyzer="word", strip_accents="unicode", lowercase=True),
    # TF-IDF sur des morceaux de 3 à 5 caractères : plus robuste aux variantes de mots
    "chars": dict(analyzer="char_wb", ngram_range=(3, 5), strip_accents="unicode", lowercase=True),
}


@dataclass
class Doc:
    id: str
    text: str


def load_corpus(path: Path = CORPUS_DIR) -> list[Doc]:
    return [Doc(p.stem, p.read_text(encoding="utf-8")) for p in sorted(path.glob("*.md"))]


def load_golden_set(path: Path = GOLDEN_SET) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TfidfRetriever:
    def __init__(self, docs: list[Doc], **vectorizer_kwargs):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(**vectorizer_kwargs)
        self.matrix = self.vectorizer.fit_transform([d.text for d in docs])

    def search(self, query: str, k: int = 3) -> list[str]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = scores.argsort()[::-1][:k]
        return [self.docs[i].id for i in order]

    def answer(self, query: str, doc_id: str) -> str:
        """Réponse extractive : la phrase du document la plus proche de la question.

        Remplace un appel LLM pour que le projet tourne sans clé API.
        """
        text = next(d.text for d in self.docs if d.id == doc_id)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip() and not s.startswith("#")]
        scores = cosine_similarity(self.vectorizer.transform([query]), self.vectorizer.transform(sentences))[0]
        return sentences[int(scores.argmax())]


def run(retriever_name: str, use_judge: bool = False) -> dict:
    docs = load_corpus()
    golden = load_golden_set()
    retriever = TfidfRetriever(docs, **RETRIEVERS[retriever_name])
    judge = None
    if use_judge:
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("--judge nécessite la variable ANTHROPIC_API_KEY (et pip install anthropic).")
        try:
            from rag_eval.judge import anthropic_judge
            judge = anthropic_judge()
        except ImportError:
            raise SystemExit("--judge nécessite le paquet anthropic : pip install anthropic")

    rows = []
    for item in golden:
        ranked = retriever.search(item["question"], k=len(docs))
        answer = retriever.answer(item["question"], ranked[0])
        row = {
            "id": item["id"],
            "question": item["question"],
            "expected_doc": item["doc"],
            "retrieved": ranked[:3],
            "recall@1": recall_at_k(ranked, item["doc"], 1),
            "recall@3": recall_at_k(ranked, item["doc"], 3),
            "rr": reciprocal_rank(ranked, item["doc"]),
            "answer": answer,
            "answer_ok": keywords_match(answer, item["keywords"]),
        }
        if judge:
            row["judge_ok"] = judge(item["question"], ", ".join(item["keywords"]), answer)
        rows.append(row)

    n = len(rows)
    summary = {
        "retriever": retriever_name,
        "n_questions": n,
        "recall@1": sum(r["recall@1"] for r in rows) / n,
        "recall@3": sum(r["recall@3"] for r in rows) / n,
        "mrr": sum(r["rr"] for r in rows) / n,
        "answer_accuracy": sum(r["answer_ok"] for r in rows) / n,
    }
    if judge:
        summary["judge_accuracy"] = sum(r["judge_ok"] for r in rows) / n
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retriever", choices=RETRIEVERS, default="words")
    parser.add_argument("--baseline", type=Path, help="run précédent (JSON) à comparer")
    parser.add_argument("--judge", action="store_true", help="ajoute un juge LLM (Anthropic)")
    parser.add_argument("--labels", type=Path, help="annotations manuelles (JSON) pour mesurer la fiabilité du contrôle automatique")
    args = parser.parse_args()

    result = run(args.retriever, use_judge=args.judge)
    s = result["summary"]
    print(f"Retriever : {s['retriever']} · {s['n_questions']} questions\n")
    print(f"  recall@1         {s['recall@1']:.0%}")
    print(f"  recall@3         {s['recall@3']:.0%}")
    print(f"  MRR              {s['mrr']:.2f}")
    print(f"  réponses justes  {s['answer_accuracy']:.0%}")
    if "judge_accuracy" in s:
        print(f"  juge LLM         {s['judge_accuracy']:.0%}")

    misses = [r for r in result["rows"] if not r["recall@1"]]
    if misses:
        print("\nMauvais document en premier :")
        for r in misses:
            print(f"  {r['id']}  {r['question']}\n        attendu {r['expected_doc']}, obtenu {r['retrieved'][0]}")

    if args.baseline:
        before = json.loads(args.baseline.read_text(encoding="utf-8"))
        diff = compare_runs(
            {r["id"]: r["answer_ok"] for r in before["rows"]},
            {r["id"]: r["answer_ok"] for r in result["rows"]},
        )
        print(f"\nComparé à {args.baseline.name} :")
        print(f"  corrections  {len(diff['fixes'])}  {' '.join(diff['fixes'])}")
        print(f"  régressions  {len(diff['regressions'])}  {' '.join(diff['regressions'])}")

    if args.labels:
        from rag_eval.judge import agreement
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
        rows = [r for r in result["rows"] if r["id"] in labels]
        human = [labels[r["id"]] for r in rows]
        checks = {"mots-clés": [r["answer_ok"] for r in rows]}
        if args.judge:
            checks["juge LLM"] = [r["judge_ok"] for r in rows]
        print(f"\nAccord avec les annotations manuelles ({len(rows)} réponses) :")
        for name, auto in checks.items():
            a = agreement(auto, human)
            wrong = [r["id"] for r, x, h in zip(rows, auto, human) if x != h]
            print(f"  {name:<10} accord {a['raw_agreement']:.0%} · kappa {a['cohen_kappa']:.2f} · désaccords {' '.join(wrong) or 'aucun'}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"rag_{args.retriever}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRun sauvegardé : {out}")


if __name__ == "__main__":
    main()

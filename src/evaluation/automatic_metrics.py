"""
Automatic metrics against reference responses (README §7.1): BLEU
(sacrebleu), BERTScore, chrF.
"""


def bleu(hypotheses: list[str], references: list[str]) -> float:
    import sacrebleu
    refs = [[r] for r in references]
    return sacrebleu.corpus_bleu(hypotheses, list(zip(*refs))).score


def chrf(hypotheses: list[str], references: list[str]) -> float:
    import sacrebleu
    refs = [[r] for r in references]
    return sacrebleu.corpus_chrf(hypotheses, list(zip(*refs))).score


def bertscore(hypotheses: list[str], references: list[str], lang: str = "en") -> dict:
    from bert_score import score as bert_score_fn
    P, R, F1 = bert_score_fn(hypotheses, references, lang=lang, verbose=False)
    return {
        "precision": float(P.mean()),
        "recall": float(R.mean()),
        "f1": float(F1.mean()),
    }


def compute_all(hypotheses: list[str], references: list[str]) -> dict:
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must be the same length")
    return {
        "n": len(hypotheses),
        "bleu": bleu(hypotheses, references),
        "chrf": chrf(hypotheses, references),
        "bertscore": bertscore(hypotheses, references),
    }


if __name__ == "__main__":
    demo_hyps = ["I dey fine, thank you", "No wahala, I go check am"]
    demo_refs = ["I dey fine o, thank you", "No wahala, make I check am"]
    import json
    print(json.dumps(compute_all(demo_hyps, demo_refs), indent=2))

"""
Code-switching metrics (README §6.4):

- Code-Mixing Index (CMI) per utterance, Gambäck & Das (2016).
- Corpus-level Multilingual Index (M-Index), after Barnett et al. (2000) as
  used by Guzmán et al. (2017) for cross-corpus comparability.
- Corpus-level Integration Index (I-Index), the mean rate of language
  alternation between adjacent tokens, after Guzmán et al. (2017).

These are approximations of the cited metrics implemented directly from
their published definitions -- useful for relative comparison across our own
data/splits/systems, not guaranteed bit-for-bit identical to any reference
implementation.
"""
from .langid import TaggedToken, tag_utterance

LANGUAGE_TAGS = ("english", "pidgin")


def cmi_utterance(text: str) -> float:
    """CMI_u = 100 * (1 - max_i(w_i) / (n - u)) if n > u else 0.

    n = total tokens, u = language-independent tokens, w_i = token count of
    language i among the non-language-independent tokens.
    """
    tags = tag_utterance(text)
    n = len(tags)
    u = sum(1 for t in tags if t.tag == "language_independent")
    if n - u <= 0:
        return 0.0
    counts: dict[str, int] = {}
    for t in tags:
        if t.tag == "language_independent":
            continue
        counts[t.tag] = counts.get(t.tag, 0) + 1
    if not counts:
        return 0.0
    max_w = max(counts.values())
    return 100.0 * (1 - max_w / (n - u))


def _corpus_language_proportions(texts: list[str]) -> dict[str, float]:
    counts = {lang: 0 for lang in LANGUAGE_TAGS}
    total = 0
    for text in texts:
        for t in tag_utterance(text):
            if t.tag in counts:
                counts[t.tag] += 1
                total += 1
    if total == 0:
        return {lang: 0.0 for lang in LANGUAGE_TAGS}
    return {lang: c / total for lang, c in counts.items()}


def multilingual_index(texts: list[str]) -> float:
    """M-Index = (1 - sum(p_i^2)) / ((k - 1) * sum(p_i^2))

    p_i = corpus-level proportion of language i tokens (language-independent
    tokens excluded), k = number of languages with p_i > 0. Ranges 0
    (monolingual) to 1 (perfectly balanced mixing).
    """
    props = _corpus_language_proportions(texts)
    present = [p for p in props.values() if p > 0]
    k = len(present)
    if k < 2:
        return 0.0
    sum_sq = sum(p * p for p in present)
    if sum_sq == 0:
        return 0.0
    return (1 - sum_sq) / ((k - 1) * sum_sq)


def integration_index(texts: list[str]) -> float:
    """I-Index = mean, over utterances, of (switch points / (n - 1)),
    where a switch point is a language change between adjacent
    non-language-independent tokens.
    """
    ratios = []
    for text in texts:
        tags = [t for t in tag_utterance(text) if t.tag != "language_independent"]
        if len(tags) < 2:
            continue
        switches = sum(
            1 for a, b in zip(tags, tags[1:]) if a.tag != b.tag
        )
        ratios.append(switches / (len(tags) - 1))
    if not ratios:
        return 0.0
    return sum(ratios) / len(ratios)


def corpus_report(texts: list[str]) -> dict:
    per_utterance = [cmi_utterance(t) for t in texts]
    return {
        "n_utterances": len(texts),
        "mean_cmi": sum(per_utterance) / len(per_utterance) if per_utterance else 0.0,
        "multilingual_index": multilingual_index(texts),
        "integration_index": integration_index(texts),
    }

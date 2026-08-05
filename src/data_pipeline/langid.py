"""
Lightweight lexicon + rule-based token-level language tagger for
English-Nigerian Pidgin text.

This is a *bootstrapping* tool, not a source of ground truth: the README
itself notes that automatic language ID is unreliable for this language
pair because their lexicons overlap heavily. Its outputs are meant to be
reviewed/corrected by `src/annotation/annotate_cli.py`, per README §6.2 step 2
and §6.4.

Tags: "english" | "pidgin" | "language_independent"
"""
import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[A-Za-z']+|\d+(?:[.,]\d+)?|[^\sA-Za-z0-9]")

# Words/particles that are either exclusively Pidgin or overwhelmingly signal
# Pidgin grammar (aspect markers, hortatives, discourse particles, common
# lexical items). Not exhaustive by design -- this is a bootstrap aid.
PIDGIN_LEXICON = {
    "abeg", "wetin", "dey", "dem", "una", "unu", "sef", "abi", "wey", "fit",
    "sabi", "don", "wahala", "waka", "chop", "palava", "comot", "jare",
    "oga", "shey", "ehen", "ehyah", "gbege", "katakata", "na", "kai", "gan",
    "ginger", "japa", "jhoor", "kolo", "sha", "o", "biko", "tey", "wella",
    "gbam", "bros", "sis", "oyinbo", "naija", "vex", "yarn", "yawa", "tush",
    "gist", "kpatakpata", "shakara", "wayo", "tori", "danfo", "okada",
    "molue", "buka", "amala", "suya", "agbero", "yeye", "nawa", "chai",
    "nko", "ahn", "abegi", "una", "unu",
}

# High-frequency function words that are ambiguous between English and
# Pidgin (shared lexicon) but default to English when not in PIDGIN_LEXICON
# and not otherwise classified.
LANGUAGE_INDEPENDENT_RE = re.compile(r"^(\d+(?:[.,]\d+)?|[^\sA-Za-z0-9]+)$")


@dataclass
class TaggedToken:
    text: str
    tag: str  # "english" | "pidgin" | "language_independent"


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def tag_token(token: str) -> str:
    if LANGUAGE_INDEPENDENT_RE.match(token):
        return "language_independent"
    lowered = token.lower()
    if lowered in PIDGIN_LEXICON:
        return "pidgin"
    return "english"


def tag_utterance(text: str) -> list[TaggedToken]:
    return [TaggedToken(tok, tag_token(tok)) for tok in tokenize(text)]


def dominant_language(tags: list[TaggedToken]) -> str | None:
    counts: dict[str, int] = {}
    for t in tags:
        if t.tag == "language_independent":
            continue
        counts[t.tag] = counts.get(t.tag, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def is_code_switched(text: str, min_minority_tokens: int = 1) -> bool:
    """A conservative genuine-mixing filter (README §6.2 step 2): requires at
    least `min_minority_tokens` tokens tagged as the non-dominant language."""
    tags = tag_utterance(text)
    counts: dict[str, int] = {}
    for t in tags:
        if t.tag == "language_independent":
            continue
        counts[t.tag] = counts.get(t.tag, 0) + 1
    if len(counts) < 2:
        return False
    minority_count = min(counts.values())
    return minority_count >= min_minority_tokens

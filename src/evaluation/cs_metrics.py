"""
Code-switching-specific evaluation (README §7.2):
  - Switching-density comparison: generated-response CMI vs. reference-
    response CMI; flags systems that under-mix (default to English).
  - Language-appropriateness rate scaffold (human-rated, see human_eval_app.py).
"""
from ..data_pipeline.cmi import cmi_utterance


def switching_density_comparison(generated: list[str], references: list[str], under_mix_threshold: float = 0.5) -> dict:
    """`under_mix_threshold`: if mean generated CMI < threshold * mean
    reference CMI, the system is flagged as under-mixing (defaulting to
    English) relative to the human references."""
    if len(generated) != len(references):
        raise ValueError("generated and references must be the same length")
    gen_cmis = [cmi_utterance(g) for g in generated]
    ref_cmis = [cmi_utterance(r) for r in references]
    mean_gen = sum(gen_cmis) / len(gen_cmis) if gen_cmis else 0.0
    mean_ref = sum(ref_cmis) / len(ref_cmis) if ref_cmis else 0.0
    under_mixing = mean_ref > 0 and mean_gen < under_mix_threshold * mean_ref
    return {
        "mean_generated_cmi": mean_gen,
        "mean_reference_cmi": mean_ref,
        "cmi_ratio": (mean_gen / mean_ref) if mean_ref else None,
        "flag_under_mixing": under_mixing,
    }


def language_appropriateness_rate(human_ratings: list[bool]) -> float:
    """`human_ratings`: one bool per response, True if a human rater judged
    the language mixture appropriate for the input (README §7.2)."""
    if not human_ratings:
        return 0.0
    return sum(human_ratings) / len(human_ratings)


if __name__ == "__main__":
    import json
    demo_generated = ["Thank you for reaching out, how can I help?", "I dey fine, thank you"]
    demo_references = ["Thank you o, wetin I fit do for you?", "I dey kampe, thank you sha"]
    print(json.dumps(switching_density_comparison(demo_generated, demo_references), indent=2))

"""Spec validation logic — stage-specific constraint checking."""

from __future__ import annotations

from pipeline.spec.schema import StrategySpec


def validate_spec(spec: StrategySpec, stage: str = "phase1") -> list[str]:
    """Validate spec against stage-specific constraints.

    Supported stages:
    - ``"phase1"``: Strict Phase I — rejects walk_forward and oos_ratio.
    - ``"phase1_statistical"``: Phase I with statistical validation — allows
      walk_forward and oos_ratio for Bootstrap CI + Walk-Forward analysis.
    - ``"phase2"``: Reserved for future use.

    Returns a list of error messages (empty list = valid).
    """
    errors: list[str] = []

    if stage == "phase1":
        if spec.validation.method == "walk_forward":
            errors.append(
                "walk_forward validation is not allowed in Phase I. "
                "Use 'full_sample' for in-sample evaluation."
            )

        if spec.validation.oos_ratio is not None:
            errors.append("oos_ratio is not allowed in Phase I (deferred to Phase II).")

    # stage="phase1_statistical" allows walk_forward/oos_ratio intentionally
    # stage="phase2" is reserved for future use

    return errors

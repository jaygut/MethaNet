"""Public-safe wording for selected internal molecular review actions.

The private projection authoring workflow imports this allowlist before emitting
case data. Unknown review cases fail closed; publication validation separately
rejects excluded internal labels in generated payloads.
"""

PUBLIC_REVIEW_ACTIONS = {
    'interpret': (
        'Review subunit phylogeny, operon context, and taxonomy; require '
        'substrate-specific validation before treating an ambiguous '
        'monooxygenase hit as methane oxidation.'
    ),
    'design': (
        'Review taxonomy and phylogeny/active sites; keep unresolved '
        'MCR-family assignments provisional.'
    ),
}


def public_fact_value(case_id, key, value):
    """Apply the public wording contract to a copied review fact."""
    if key != 'Review action':
        return value
    try:
        return PUBLIC_REVIEW_ACTIONS[case_id]
    except KeyError as exc:
        raise ValueError(f'No public review-action wording for case {case_id!r}.') from exc

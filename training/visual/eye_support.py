"""Input-only upper bounds, never model predictions or threshold selection."""


def summarize_support(windows):
    def group(rows):
        positives=[r for r in rows if r['label_known'] and r['target']==1]
        counts=lambda selected:dict(positive=len(selected),
            endpoint_available=sum(bool(r['eye_endpoint_available']) for r in selected),
            duration_eligible=sum(bool(r['eye_duration_eligible']) for r in selected))
        return dict(overall=counts(positives),
            by_subtype={s:counts([r for r in positives if r['subtype']==s])
                        for s in ('closed_eye','yawn','overlap','unresolved')},
            negative_duration_eligible=sum(bool(r['eye_duration_eligible']) for r in rows if r['label_known'] and r['target']==0))
    return dict(group(windows),by_subject={s:group([r for r in windows if r['subject_id']==s])
                    for s in sorted({r['subject_id'] for r in windows})},
        meaning='All available eye pixels hypothetically closed; causal timing and missing/reset policy unchanged. Not model accuracy.')


def require_eye_support(report):
    pure=report['by_subtype']['closed_eye']
    target=pure if pure['positive'] else report['overall']
    if not target['positive'] or not target['duration_eligible']:
        raise ValueError('No continuous eye support for positive windows; no training started. '
                         'Inspect eye-support.json and eye-window-coverage.csv, then repair extraction first.')

"""Read-only eligibility audit; reports input coverage, never predicted accuracy."""
import argparse
import csv
from pathlib import Path

from training.common.checkpoint import atomic_json, load_checkpoint
from training.common.data import file_sha256
from training.gaze.checkpoint import validate_metadata
from training.gaze.data import FeatureCorpus
from training.gaze.head_evidence import head_statistics
from training.gaze.head_policy import minimum_head_count


POLICIES = {
    'strict': dict(min_valid_ratio=.5, max_gap_steps=0),
    'coverage25': dict(min_valid_ratio=.25, max_gap_steps=0),
    'gap80': dict(min_valid_ratio=.5, max_gap_steps=2),
}


def _counts():
    return dict(windows=0, positive=0, negative=0, unknown=0, eligible=0,
                eligible_positive=0, eligible_negative=0, eligible_unknown=0,
                newly_eligible=0, newly_eligible_positive=0,
                newly_eligible_negative=0, newly_eligible_unknown=0)


def audit_coverage(checkpoint, *, output_dir, manifest=None, data_root=None,
                   split='val', view='body', subset='distraction', progress=None):
    if split not in ('train', 'val', 'test') or view not in ('body', 'face', 'all') or subset not in ('distraction', 'all'):
        raise ValueError('Invalid split/view/subset selection')
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Use a NEW output_dir; existing reports are never overwritten')
    payload = load_checkpoint(checkpoint)
    validate_metadata(payload)
    config = payload['config']
    manifest = manifest or config['runtime']['manifest']
    data_root = data_root or config['runtime']['data_root']
    if not manifest:
        raise ValueError('Supply --manifest; checkpoint has no saved manifest path')
    if progress: progress('Checking original feature hashes and effective head masks...')
    corpus = FeatureCorpus(config, manifest, data_root)
    try:
        if corpus.summary != payload['data_summary']:
            raise ValueError('Data fingerprint/provenance differs from the checkpoint corpus')
        dataset = corpus.dataset(split, payload['normalizer'])
        indices = [i for i in range(len(dataset))
                   if (view == 'all' or dataset.metadata(i)['view'] == view)
                   and (subset == 'all' or dataset.metadata(i)['subset'] == subset)]
        if not indices: raise ValueError('No windows match the selection')
        variants = {name: {} for name in POLICIES}
        gap_bins = {}
        output.mkdir(parents=True, exist_ok=False)
        with (output/'windows.csv').open('x', encoding='utf-8-sig', newline='') as stream:
            writer = None
            for order, i in enumerate(indices):
                inputs, target, labels, _, _ = dataset[i]
                record, start = dataset.locate(i)
                stats = head_statistics(inputs['x'])
                gap, ratio = int(stats['gap_steps']), float(stats['valid_ratio'])
                known = bool(labels[3])
                category = 'unknown' if not known else 'positive' if target[3] else 'negative'
                # Each count represents a window, not an independent driving event.
                bin_name = 'none' if gap == dataset.steps else str(gap) if gap <= 2 else '3-5' if gap <= 5 else '6-25' if gap <= 25 else '>25'
                bins = gap_bins.setdefault(record['view'], {})
                bins.setdefault(bin_name, dict(positive=0, negative=0, unknown=0))[category] += 1
                row = {key: record[key] for key in ('record_id', 'recording_id', 'subject_id', 'view', 'subset', 'split')}
                row.update(window_index=i, start_slot=start, end_slot=start+dataset.steps-1,
                           window_start_s=start/config['data']['sample_rate_hz'],
                           window_end_s=(start+dataset.steps-1)/config['data']['sample_rate_hz'],
                           label_known=known, target=int(target[3]) if known else None,
                           h_valid_ratio=ratio, h_current_valid=bool(stats['current_valid']),
                           h_gap_steps=gap if gap < dataset.steps else None,
                           h_gap_ms=gap*1000/config['data']['sample_rate_hz'] if gap < dataset.steps else None)
                for name, policy in POLICIES.items():
                    eligible = gap <= policy['max_gap_steps'] and stats['valid_count'] >= minimum_head_count(policy, dataset.steps)
                    newly = eligible and not bool(stats['eligible'])
                    row[name+'_eligible'] = eligible
                    counts = variants[name].setdefault(record['view'], _counts())
                    counts['windows'] += 1
                    counts[category] += 1
                    counts['eligible'] += int(eligible)
                    counts['eligible_'+category] += int(eligible)
                    counts['newly_eligible'] += int(newly)
                    counts['newly_eligible_'+category] += int(newly)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                if progress and (order % 512 == 0 or order+1 == len(indices)):
                    progress(f'Audited {order+1}/{len(indices)} windows')
        report = dict(schema_version='driveguard-head-coverage-v1', status='complete',
                      checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=file_sha256(checkpoint),
                      checkpoint_epoch=payload['epoch'], model_kind=payload['model_kind'],
                      data_fingerprint=corpus.summary['fingerprint'],
                      manifest=str(Path(manifest).resolve()), data_root=str(corpus.root),
                      synthetic=corpus.manifest['extractor_profile'].get('synthetic', False),
                      selection=dict(split=split, view=view, subset=subset),
                      policies=POLICIES, variants=variants, gap_bins=gap_bins,
                      notes=['No model inference, training, normalizer fitting, or cache edits were performed.',
                             'New eligibility is relative to strict (.5 coverage, zero gap), not checkpoint predictions.',
                             'Eligibility counts are not recall or predicted recoveries; negative windows can add false positives.',
                             'Gap milliseconds measure sample-grid intervals, not exact source measurement age.',
                             'No valid head anywhere in a window is reported as null gap and the none bin.',
                             'Labels only stratify the report and never determine eligibility.'])
        atomic_json(output/'summary.json', report)
        return report
    finally:
        corpus.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--data-root')
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    parser.add_argument('--view', choices=('body', 'face', 'all'), default='body')
    parser.add_argument('--subset', choices=('distraction', 'all'), default='distraction')
    args = parser.parse_args(argv)
    try:
        report = audit_coverage(**vars(args), progress=lambda message: print(message, flush=True))
        print(f"Complete: {Path(args.output_dir)/'summary.json'} ({len(report['policies'])} policies)")
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__': raise SystemExit(main())

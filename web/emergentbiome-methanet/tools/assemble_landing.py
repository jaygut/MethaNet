#!/usr/bin/env python3
"""Stage a reviewed landing-only release without touching the report alias.

Inputs: this website's runtime files and a local publication-review receipt.
Output: a NEW staging directory plus a separate checksum manifest.
No network requests, deletion, git writes or deployment are performed here.
"""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import urlsplit

WEB = Path(__file__).resolve().parents[1]
ITEMS = ('index.html', 'styles.css', 'molecular-applications.css',
         'molecular-network.css', 'main.js', 'config.js', 'lib', 'scenes',
         'vendor', 'data', 'assets', 'CNAME')
DATA_FILES = {'atlas.json', 'molecular-application-cases-public-v1.json',
              'molecular-evidence-network-public-v1.json'}
REVIEWED_DATA_FILES = {
    'data/molecular-application-cases-public-v1.json',
    'data/molecular-evidence-network-public-v1.json',
}
REQUIRED_REVIEWERS = {'science_provenance', 'public_boundary', 'operations_portability'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_dependencies(root):
    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.paths = set()

        def handle_starttag(self, tag, attrs):
            values = dict(attrs)
            if tag in ('script', 'img') and values.get('src'):
                self.paths.add(values['src'])
            if tag == 'link' and values.get('href'):
                self.paths.add(values['href'])

    parser = Links()
    parser.feed((root / 'index.html').read_text())
    return sorted(urlsplit(p).path for p in parser.paths
                  if not urlsplit(p).scheme and not p.startswith('//'))


def validate(root, source=False):
    missing = [path for path in local_dependencies(root) if not (root / path).is_file()]
    if missing:
        raise ValueError('Missing runtime dependencies: ' + ', '.join(missing))
    actual = set(p.name for p in (root / 'data').iterdir())
    if (not DATA_FILES.issubset(actual)) or (not source and actual != DATA_FILES):
        raise ValueError('Unreviewed data files or missing required data files.')
    if (root / 'CNAME').read_text().strip() != 'emergentbiome.earth':
        raise ValueError('Custom-domain binding changed.')
    if '<meta name="robots" content="noindex"' not in (root / 'index.html').read_text():
        raise ValueError('The controlled-diligence indexing boundary must remain intact.')
    for name in DATA_FILES - {'atlas.json'}:
        content = (root / 'data' / name).read_text()
        data = json.loads(content)
        if data.get('publication_status') != 'scoped_public_demonstration':
            raise ValueError(name + ' has no scoped public presentation release.')
        if not data.get('public_sources'):
            raise ValueError(name + ' lacks public source attribution.')
        if re.search(r'\bK\d{5}\b|\bV17\b|sourcePayload|kegg_hit|pfam_hits|/home/|results/|/policy/|internal_review|unknown_no_external_export', content):
            raise ValueError(name + ' contains excluded annotation or internal material.')
    return {'local_dependencies': len(local_dependencies(root)), 'data_files': sorted(DATA_FILES)}


def validate_publication_review(review_path, root=WEB, require_tracked=False):
    """Require a current, hash-bound approval for both public evidence files."""
    clearance = json.loads(Path(review_path).read_text())
    if clearance.get('decision') != 'approved_scoped_landing':
        raise ValueError('Publication review has not approved this bounded landing release.')
    if clearance.get('scope') != 'three_case_presentation_only':
        raise ValueError('Unexpected publication scope.')
    reviewers = clearance.get('reviewers', {})
    if (set(reviewers) != REQUIRED_REVIEWERS
            or any(status != 'approved' for status in reviewers.values())):
        raise ValueError('All three required review lanes must approve this release.')
    expected = clearance.get('files', {})
    if set(expected) != REVIEWED_DATA_FILES:
        raise ValueError('Review must bind both public case-data projections and no other files.')
    if any(not (root / name).is_file() or digest(root / name) != value
           for name, value in expected.items()):
        raise ValueError('Reviewed presentation hashes do not match the current files.')
    if require_tracked:
        repository = root.resolve().parents[1]
        receipt_relative = Path(review_path).resolve().relative_to(repository)
        data_paths = [(root / name).resolve().relative_to(repository)
                      for name in sorted(REVIEWED_DATA_FILES)]
        controlled_files = [str(receipt_relative), *(str(path) for path in data_paths)]
        for name in controlled_files:
            tracked = subprocess.run(
                ['git', '-C', str(repository), 'ls-files', '--error-unmatch', '--', name],
                capture_output=True, text=True, check=False,
            )
            if tracked.returncode:
                raise ValueError(f'Publication approval input is not tracked by Git: {name}')
        changed = subprocess.run(
            ['git', '-C', str(repository), 'diff', '--quiet', 'HEAD', '--', *controlled_files],
            check=False,
        )
        if changed.returncode:
            raise ValueError('Publication approval inputs must be committed before deployment.')
    return {'decision': clearance['decision'], 'scope': clearance['scope'],
            'approved_reviewers': sorted(REQUIRED_REVIEWERS),
            'reviewed_files': sorted(REVIEWED_DATA_FILES)}


def assemble(destination, review, manifest):
    destination = destination.resolve()
    if destination.exists():
        raise ValueError('Use a new staging directory; existing files are preserved.')
    validate_publication_review(review)
    checked = validate(WEB, source=True)
    destination.mkdir(parents=True)
    for item in ITEMS:
        src, dst = WEB / item, destination / item
        if item == 'data':
            dst.mkdir()
            for name in sorted(DATA_FILES):
                shutil.copy2(src / name, dst / name)
        elif src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    (destination / '.nojekyll').touch()
    validate(destination)
    files = {str(p.relative_to(destination)): digest(p)
             for p in sorted(destination.rglob('*')) if p.is_file()}
    if any(p.startswith('report/') for p in files):
        raise ValueError('A landing-only stage must preserve the independently published report.')
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({'scope': 'landing_only', 'files': files,
        'publication_review_sha256': digest(review), 'checks': checked}, indent=2) + '\n')
    return {'files': len(files), 'bytes': sum((destination / p).stat().st_size for p in files)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-root', type=Path)
    parser.add_argument('--validate-publication-review', type=Path)
    parser.add_argument('--require-tracked-review', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--publication-review', type=Path)
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    if args.validate_root:
        print(json.dumps(validate(args.validate_root.resolve())))
    elif args.validate_publication_review:
        print(json.dumps(validate_publication_review(
            args.validate_publication_review.resolve(),
            require_tracked=args.require_tracked_review,
        )))
    elif args.output_dir and args.publication_review and args.manifest:
        print(json.dumps(assemble(args.output_dir, args.publication_review, args.manifest)))
    else:
        parser.error('Supply --validate-root or all three staging arguments.')

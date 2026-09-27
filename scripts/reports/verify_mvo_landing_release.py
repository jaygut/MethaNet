#!/usr/bin/env python3
"""Verify every reviewed runtime byte, the preserved report and excluded URLs.

Read-only HTTP acceptance for an assembled or live landing release. The caller
supplies a local manifest, expected report digest and an explicit output receipt.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--report-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    token = hashlib.sha256(manifest_bytes).hexdigest()[:16]

    def fetch(item):
        path, expected = item
        url = args.base.rstrip('/') + '/' + quote(path, safe='/') + '?release=' + token
        try:
            with urlopen(Request(url, headers={'Cache-Control': 'no-cache'}), timeout=45) as response:
                body, status = response.read(), response.status
        except HTTPError as error:
            body, status = error.read(), error.code
        actual = hashlib.sha256(body).hexdigest()
        return {'path': path, 'status': status, 'expected_sha256': expected,
                'sha256': actual if status == 200 else None,
                'pass': status == (404 if expected is None else 200)
                and (expected is None or actual == expected)}

    items = list(manifest['files'].items())
    items += [('report/index.html', args.report_sha256),
              ('data/molecular-evidence-network-v1.json', None),
              ('data/molecular-application-cases-v1.json', None)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        checks = list(pool.map(fetch, items))
    result = {'checked_at': datetime.now(timezone.utc).isoformat(),
              'base': args.base, 'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
              'checks': checks, 'passed': all(check['pass'] for check in checks)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'base': args.base, 'checks': len(checks),
                      'failures': [check for check in checks if not check['pass']]}))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()

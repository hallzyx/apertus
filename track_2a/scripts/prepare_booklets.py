"""Fetch source PDFs and derive reproducible full-booklet evaluation inputs."""
import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


def main():
    from ost_nli.data import load_dataset, fingerprint
    from ost_nli.documents import pdf_document, reference_coverage
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--splits', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--pdf-dir', help='Reuse only original PDF downloads, regenerate processed documents')
    args = p.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    raw = [json.loads(s) for s in Path(args.source).read_text().splitlines() if s.strip()]
    urls = sorted({r['booklet_url'] for r in raw})
    expected_path=Path(__file__).resolve().parents[1]/'experiments/booklet-source-v2/manifest.json'
    expected={r['url']:r['sha256'] for r in json.loads(expected_path.read_text())['records']
              if r.get('status')=='completed'} if expected_path.exists() else {}
    def fetch(url):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != 'https' or parsed.hostname != 'www.bk.admin.ch':
            raise ValueError('Unexpected official source URL')
        key = hashlib.sha256(url.encode()).hexdigest()
        pdf = (Path(args.pdf_dir) if args.pdf_dir else root) / (key + '.pdf'); target = root / (key + '.json')
        error = None
        for attempt in range(3):
            try:
                if pdf.exists() and url in expected and hashlib.sha256(pdf.read_bytes()).hexdigest()!=expected[url]:
                    # Preserve corrupt downloads for diagnosis; never accept or reuse them.
                    backup=pdf.with_name(pdf.name+'.rejected-'+str(time.time_ns()))
                    pdf.replace(backup)
                if not pdf.exists():
                    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        content = response.read(50_000_001)
                    if len(content) > 50_000_000 or not content.startswith(b'%PDF-'):
                        raise ValueError('Invalid or oversized PDF')
                    if url in expected and hashlib.sha256(content).hexdigest()!=expected[url]:
                        raise ValueError('Downloaded PDF does not match pinned source checksum')
                    partial = pdf.with_suffix('.part'); partial.write_bytes(content); partial.replace(pdf)
                if target.exists():
                    doc = json.loads(target.read_text())
                    if doc['source_sha256'] != hashlib.sha256(pdf.read_bytes()).hexdigest():
                        raise ValueError('Cached source PDF changed')
                    if doc['processing']['version'] != 'page-offset-plain-v2':
                        raise ValueError('Cached processing version mismatch; use a new output directory')
                else:
                    doc = pdf_document(pdf, url)
                target.write_text(json.dumps(doc, ensure_ascii=False) + '\n')
                return {'url': url, 'pdf': pdf.name, 'document': target.name,
                        'sha256': doc['source_sha256'], 'bytes': pdf.stat().st_size,
                        'pages': doc['pages'], 'passages': len(doc['passages']), 'status': 'completed'}
            except Exception as exc:
                error = type(exc).__name__ + ': ' + str(exc)[:160]
                if attempt < 2: time.sleep(1 + attempt)
        return {'url': url, 'status': 'failed', 'error': error}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = []
        for record in pool.map(fetch, urls):
            records.append(record); print(record['status'], record['url'], flush=True)
    manifest = {'source_sha256': hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
                'records': records, 'document_processing': 'page-offset-plain-v2',
                'reference_used_for_production_context': False}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    by_url = {r['url']: r for r in records if r['status'] == 'completed'}
    documents = {u: json.loads((root / r['document']).read_text()) for u, r in by_url.items()}
    audit = {'booklets_total': len(urls), 'booklets_downloaded': len(by_url), 'partitions': {}}
    for name in ['train', 'validation', 'test']:
        rows = load_dataset(Path(args.splits) / (name + '.jsonl'))
        full, diagnostics, missing = [], [], []
        for row in rows:
            url = row['source_booklet_url']
            if url not in documents:
                missing.append(row['id']); continue
            doc = documents[url]
            reference = '\n\n'.join(p['text'] for p in row['passages'])
            full.append({**row, 'document_id': doc['document_id'], 'passages': doc['passages'],
                         'document_scope': 'full_booklet', 'source_pdf_sha256': doc['source_sha256'],
                         'evidence_ids': [], 'provided_reference_ids': [],
                         'evidence_annotation_available': False})
            diagnostics.append({'id': row['id'], 'reference_5gram_coverage_in_full_booklet':
                                reference_coverage(doc['passages'], reference)})
        (root / (name + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in full))
        audit['partitions'][name] = {'n': len(full), 'missing_ids': missing, 'sha256': fingerprint(full),
                                     'reference_alignment': diagnostics}
    audit['note'] = 'Reference 5-gram coverage is diagnostic alignment, not organizer gold evidence scoring. No missing rows silently scored.'
    (root / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print('Completed booklets:', len(by_url), '/', len(urls), flush=True)


if __name__ == '__main__': main()

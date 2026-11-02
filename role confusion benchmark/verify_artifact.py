"""Verify the saved experiment without PyTorch, internet access, or a GPU."""
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from build_and_run import generate, score

root = Path(__file__).resolve().parent
manifest = json.loads((root/'manifest.json').read_text())
for name, expected in manifest['sha256'].items():
    actual = hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'Checksum mismatch: {name}')

folder = root/'results/pilot-20261102'
cases = [json.loads(x) for x in (folder/'cases.jsonl').read_text().splitlines()]
assert cases == generate()
assert (folder/'cases.jsonl').read_bytes() == (root/'addbench_pilot.jsonl').read_bytes()
assert len(cases) == 48 and len({r['id'] for r in cases}) == 48
revisions = json.loads((root/'models.lock.json').read_text())
total = 0
with tempfile.TemporaryDirectory() as temporary:
    scratch = Path(temporary)
    for model, revision in revisions.items():
        name = model.split('/')[-1]
        path = folder/(name+'.jsonl')
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        assert [r['id'] for r in rows] == [r['id'] for r in cases]
        metadata = json.loads((folder/(name+'.metadata.json')).read_text())
        assert metadata['model_revision'] == revision
        assert metadata['do_sample'] is False and metadata['max_new_tokens'] == 96
        for case, row in zip(cases, rows):
            assert row['model'] == model
            assert all(row[k] == case[k] for k in ['task','source','phrasing','expected_a','forbidden_b','b_marker'])
            assert all(row[k] == v for k,v in score(case,row['output']).items())
            assert row['rendered_prompt'] and not row['reached_token_limit']
        (scratch/path.name).write_bytes(path.read_bytes())
        total += len(rows)
    subprocess.run([sys.executable,str(root/'analyze_results.py'),str(scratch)],check=True,capture_output=True)
    with (folder/'summary.csv').open(newline='') as f:
        expected = list(csv.DictReader(f))
    with (scratch/'summary.csv').open(newline='') as f:
        actual = list(csv.DictReader(f))
    assert actual == expected
assert total == 144
print('PASS: file checksums, exact 48-case dataset, 144 saved responses, revisions, scores, and summary CSV.')
print('This verifies saved artifacts; it does not rerun model inference.')

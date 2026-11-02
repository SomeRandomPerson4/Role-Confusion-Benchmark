"""Run the unchanged pilot on three small open-weight models, sequentially."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

MODELS = ['Qwen/Qwen3-0.6B', 'Qwen/Qwen2.5-0.5B-Instruct',
          'HuggingFaceTB/SmolLM2-360M-Instruct']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out-dir', type=Path, default=Path('results/pilot'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.out_dir / 'cases.jsonl'
    if dataset.exists():
        raise SystemExit('Choose a new --out-dir; this directory already contains a run.')
    dataset.write_bytes((root / 'addbench_pilot.jsonl').read_bytes())
    statuses = []
    for model in MODELS:
        slug = model.split('/')[-1]
        command = [sys.executable, '-u', str(root / 'build_and_run.py'),
                   '--input', str(dataset.resolve()), '--model', model,
                   '--device', args.device, '--max-new-tokens', '96',
                   '--out', str((args.out_dir / (slug + '.jsonl')).resolve())]
        print('Running', model, flush=True)
        with (args.out_dir / (slug + '.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        statuses.append(dict(model=model,exit_code=result.returncode,command=command))
        (args.out_dir / 'status.json').write_text(json.dumps(statuses, indent=2))
        print(model, 'exit code', result.returncode, flush=True)
    if any(s['exit_code'] for s in statuses):
        raise SystemExit('Some models failed; inspect status.json and model logs.')

if __name__ == '__main__':
    main()

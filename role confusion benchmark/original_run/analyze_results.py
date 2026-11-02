"""Summarize pilot metrics without treating marker detection as semantic grading."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    for path in sorted(args.directory.glob('*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if 'output' in row:
                groups[(row['model'],row['source'],row['phrasing'],row['role_serialization'])].append(row)
    fields = ['model','source','phrasing','role_serialization','n','a_correct',
              'scope_violation','explicit_b','a_correct_and_explicit_b','exact_compliance','token_limit_count']
    output = args.directory / 'summary.csv'
    with output.open('w',newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, rows in sorted(groups.items()):
            record = dict(zip(fields[:4],key))
            record['n'] = len(rows)
            for metric in ['a_correct','scope_violation','explicit_b','exact_compliance']:
                record[metric] = sum(bool(r[metric]) for r in rows)/len(rows)
            record['a_correct_and_explicit_b'] = sum(r['a_correct'] and r['explicit_b'] for r in rows)/len(rows)
            record['token_limit_count'] = sum(r.get('reached_token_limit',False) for r in rows)
            writer.writerow(record)
    print(output)

if __name__ == '__main__':
    main()

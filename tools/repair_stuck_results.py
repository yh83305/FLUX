"""Freeze selection, recoverably retire deadlock artifacts, publish reruns by ID."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
from collections import defaultdict


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_under(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def stuck(row):
    return (float(row['success']) == 0 and float(row['collision']) == 0
            and abs(float(row['time_to_goal']) - 4.5) < .01
            and abs(float(row['trajectory_length'])) < .1)


def plan(source, replacement, target):
    source, replacement = source.resolve(), replacement.resolve()
    assert source != replacement
    records, ids = [], {}
    for model in ['iplanner', 'viplanner']:
        ids[model] = {}
        for scene in ['Full_Warehouse', 'Hospital', 'Office']:
            folder = source/model/scene
            marks = sorted(folder.glob('episode_*/COMPLETE.json'))
            assert len(marks) == 100, folder
            ids[model][scene] = []
            for mark in marks:
                data = json.loads(mark.read_text())
                ep = int(data['episode'])
                result = Path(data['result_dir']).resolve()
                assert is_under(result, folder.resolve())
                rows = [r for r in csv.DictReader((result/'metric.csv').open()) if int(r['episode']) == ep]
                assert len(rows) == 1
                if not stuck(rows[0]):
                    continue
                assert rows[0] == data['metric']
                ids[model][scene].append(ep)
                records.append(dict(model=model, scene=scene, episode=ep,
                                    marker=str(mark), marker_sha256=digest(mark),
                                    result_dir=str(result), metric=rows[0],
                                    csv_sha256=digest(result/'metric.csv')))
    write(target, dict(source=str(source), replacement=str(replacement),
                       archive=str(source/('_retired_stuck_'+replacement.name)),
                       criterion='success=0, pedestrian collision=0, abs(duration-4.5)<.01s, abs(path)<.1m',
                       protocol='Only selected startup-deadlock cases rerun with lookahead MPC; retained rows use native MPC. This is a mixed-controller repair, not a uniform-controller evaluation.',
                       episode_ids=ids, records=records))
    print(json.dumps(dict(count=len(records), episode_ids=ids)))


def retire(p):
    source, archive = Path(p['source']), Path(p['archive'])
    assert is_under(archive, source) and archive.name.startswith('_retired_stuck_')
    groups = defaultdict(list)
    for row in p['records']:
        assert row['model'] in ('iplanner', 'viplanner') and stuck(row['metric'])
        result = Path(row['result_dir'])
        assert is_under(result, source/row['model']/row['scene'])
        groups[result].append(row)
        mark = Path(row['marker'])
        oldmark = archive/'markers'/mark.relative_to(source)
        assert digest(mark if mark.exists() else oldmark) == row['marker_sha256']
    # Preflight and snapshot each shared CSV before removing selected rows.
    for result, rows in groups.items():
        backup = archive/'raw'/result.relative_to(source)/'metric.csv'
        expected = {r['csv_sha256'] for r in rows}
        assert len(expected) == 1
        assert digest(backup if backup.exists() else result/'metric.csv') in expected
    for result in groups:
        backup = archive/'raw'/result.relative_to(source)/'metric.csv'
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(result/'metric.csv', backup)
    # Invalidating completion is recoverable, as are all episode artifacts.
    affected = {source, *(source/r['model'] for r in p['records']),
                *(source/r['model']/r['scene'] for r in p['records'])}
    for folder in affected:
        for name in ['COMPLETE', 'status.json']:
            old = folder/name
            backup = archive/'metadata'/old.relative_to(source)
            if old.exists() and not backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(old, backup)
            if name == 'COMPLETE' and old.exists():
                old.unlink()
        write(folder/'status.json', dict(state='pending_selected_mpc_repair', repair_manifest=p['replacement']+'/repair_plan.json'))
    for row in p['records']:
        mark = Path(row['marker'])
        oldmark = archive/'markers'/mark.relative_to(source)
        if mark.exists():
            oldmark.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(mark), str(oldmark))
        for name in [f"fps_{row['episode']}.mp4", f"first_frame_{row['episode']}.png"]:
            file = Path(row['result_dir'])/name
            dest = archive/'raw'/file.relative_to(source)
            if file.exists():
                assert not dest.exists(), dest
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(file), str(dest))
    for result, rows in groups.items():
        backup = archive/'raw'/result.relative_to(source)/'metric.csv'
        reader = csv.DictReader(backup.open())
        selected = {r['episode'] for r in rows}
        retained = [r for r in reader if int(r['episode']) not in selected]
        tmp = result/'metric.csv.repair_tmp'
        with tmp.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
            writer.writeheader(); writer.writerows(retained)
        tmp.replace(result/'metric.csv')
    write(source/'SELECTED_MPC_REPAIR.json', dict(state='pending', manifest=p['replacement']+'/repair_plan.json', protocol=p['protocol']))
    write(Path(p['replacement'])/'RETIRE_COMPLETE.json', dict(count=len(p['records']), archive=str(archive)))
    print('Retired recoverably:', len(p['records']))


def publish(p):
    source, replacement = Path(p['source']), Path(p['replacement'])
    new_records = []
    for row in p['records']:
        mark = replacement/row['model']/row['scene']/f"episode_{row['episode']:03d}"/'COMPLETE.json'
        record = json.loads(mark.read_text())
        assert record['episode'] == row['episode']
        result = Path(record['result_dir']).resolve()
        assert is_under(result, replacement)
        assert json.loads((result/'controller_config.json').read_text())['selection'] == 'lookahead'
        rows = [r for r in csv.DictReader((result/'metric.csv').open()) if int(r['episode']) == row['episode']]
        assert len(rows) == 1 and rows[0] == record['metric']
        assert (result/f"fps_{row['episode']}.mp4").stat().st_size > 1000
        record['repair'] = dict(controller='lookahead', supersedes=str(Path(p['archive'])/'markers'/Path(row['marker']).relative_to(source)), protocol=p['protocol'])
        new_records.append((Path(row['marker']), record))
    for mark, record in new_records:
        if mark.exists():
            assert json.loads(mark.read_text()) == record
        else:
            write(mark, record)
    for model, scenes in p['episode_ids'].items():
        for scene in scenes:
            folder = source/model/scene
            records = [json.loads(f.read_text()) for f in folder.glob('episode_*/COMPLETE.json')]
            assert len(records) == 100 and {r['episode'] for r in records} == set(range(100))
            write(folder/'status.json', dict(state='complete_mixed_controllers', completed=100, repaired_ids=scenes[scene]))
            (folder/'COMPLETE').touch()
        write(source/model/'status.json', dict(state='complete_mixed_controllers', episodes=300))
        (source/model/'COMPLETE').touch()
    write(source/'SELECTED_MPC_REPAIR.json', dict(state='complete', manifest=str(replacement/'repair_plan.json'), protocol=p['protocol']))
    if all((source/m/'COMPLETE').exists() for m in ['ours_grpo_rule', 'iplanner', 'viplanner']):
        write(source/'status.json', dict(state='complete_mixed_controllers', episodes=900, protocol=p['protocol']))
        (source/'COMPLETE').touch()
    write(replacement/'PUBLISHED.json', dict(replaced=len(new_records), protocol=p['protocol']))
    print('Published replacements:', len(new_records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['plan', 'retire', 'publish'])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--replacement', type=Path)
    args = parser.parse_args()
    if args.action == 'plan':
        assert not args.manifest.exists()
        plan(args.source, args.replacement, args.manifest)
    else:
        globals()[args.action](json.loads(args.manifest.read_text()))

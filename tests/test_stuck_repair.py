import csv
import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location('repair', Path(__file__).parents[1]/'tools/repair_stuck_results.py')
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def test_retirement_is_recoverable_and_publish_preserves_other_rows(tmp_path):
    source, replacement = tmp_path/'source', tmp_path/'rerun'
    fields = ['episode', 'success', 'collision', 'time_to_goal', 'trajectory_length']
    for model in ['iplanner', 'viplanner']:
        for scene in ['Full_Warehouse', 'Hospital', 'Office']:
            folder = source/model/scene
            result = folder/'continuous_000'
            result.mkdir(parents=True)
            rows = [dict(zip(fields, map(str, [i, 0 if i == 7 else 1, 0, 4.5 if i == 7 else 10, 0 if i == 7 else 5]))) for i in range(100)]
            with (result/'metric.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            for row in rows:
                ep = int(row['episode'])
                repair.write(folder/f'episode_{ep:03d}'/'COMPLETE.json', dict(episode=ep, result_dir=str(result), metric=row))
            (result/'fps_7.mp4').write_bytes(b'x'*2000)
            (folder/'COMPLETE').touch()
    manifest = replacement/'repair_plan.json'
    repair.plan(source, replacement, manifest)
    p = json.loads(manifest.read_text())
    assert len(p['records']) == 6
    repair.retire(p)
    repair.retire(p)  # Interrupted cleanup can be resumed without losing backups.
    for record in p['records']:
        mark = Path(record['marker'])
        assert not mark.exists()
        assert (mark.parent.parent/'episode_008/COMPLETE.json').exists()
        assert len(list(csv.DictReader((Path(record['result_dir'])/'metric.csv').open()))) == 99
        folder = replacement/record['model']/record['scene']
        result = folder/'continuous_007'
        result.mkdir(parents=True)
        row = dict(record['metric'], success='1', time_to_goal='60', trajectory_length='29')
        with (result/'metric.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerow(row)
        (result/'fps_7.mp4').write_bytes(b'x'*2000)
        repair.write(result/'controller_config.json', dict(selection='lookahead'))
        repair.write(folder/'episode_007/COMPLETE.json', dict(episode=7, result_dir=str(result), metric=row))
    repair.publish(p)
    repair.publish(p)
    assert (replacement/'PUBLISHED.json').exists()
    assert json.loads(Path(p['records'][0]['marker']).read_text())['metric']['success'] == '1'
    assert len(list(Path(p['archive']).rglob('fps_7.mp4'))) == 6

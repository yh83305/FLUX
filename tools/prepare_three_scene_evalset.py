"""Freeze native Full Warehouse + revised Hospital/Office episode JSONs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p = argparse.ArgumentParser()
p.add_argument("--native-scenes", type=Path, required=True)
p.add_argument("--revised", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--ours-manifest", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
records, scenes = [], {}
revised_manifest = json.loads((a.revised/"manifest.json").read_text())
revised_hashes = {(r["scene"], r["episode"]): r["sha256"] for r in revised_manifest["records"]}
for name in ("Full_Warehouse", "Hospital", "Office"):
    source = a.native_scenes/name if name == "Full_Warehouse" else a.revised/name
    folder = a.output/name
    folder.mkdir()
    counts, signatures = set(), set()
    for episode in range(100):
        src = source/f"episode_{episode}.json"
        content = src.read_bytes()
        sha = hashlib.sha256(content).hexdigest()
        if name != "Full_Warehouse":
            assert sha == revised_hashes[(name, episode)]
        data = json.loads(content)["episode"]
        assert int(data["episode_id"]) == episode
        people = int(data["characters"]["num_characters"])
        assert people == len(data["characters"]["spawn_positions"])
        counts.add(people)
        semantic = dict(data)
        semantic.pop("episode_id")
        signatures.add(hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest())
        dst = folder/src.name
        shutil.copy2(src, dst)
        assert hashlib.sha256(dst.read_bytes()).hexdigest() == sha
        preview = source/f"episode_{episode}.png"
        if name != "Full_Warehouse" and preview.exists():
            shutil.copy2(preview, folder/preview.name)
        records.append(dict(scene=name, episode=episode, sha256=sha,
                            source=str(src.resolve()), people=people))
    assets = a.native_scenes/name
    for asset in assets.iterdir():
        if asset.name.startswith("episode_") or asset.name == "pointgoal_start_goal_pairs.npy":
            continue
        (folder/asset.name).symlink_to(asset.resolve(), target_is_directory=asset.is_dir())
    view = a.output/"scene_views"/name
    view.mkdir(parents=True)
    (view/name).symlink_to(Path("../..")/name, target_is_directory=True)
    scenes[name] = dict(protocol="native" if name=="Full_Warehouse" else "quality100_p10",
                        episodes=100, people=sorted(counts), semantic_unique=len(signatures),
                        source=str(source.resolve()))
ours = json.loads(a.ours_manifest.read_text())
checkpoint = Path(ours["models"][0]["checkpoint"])
h = hashlib.sha256()
with checkpoint.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024*1024), b""):
        h.update(chunk)
assert h.hexdigest() == ours["checkpoint_sha256"]
shutil.copy2(a.ours_manifest, a.output/"ours_reference_manifest.json")
selector = Path(ours["repo"])/"CrowdSim/config/eval/navdp_modeless_esdf.yaml"
shutil.copy2(selector, a.output/"ours_selector_config.yaml")
manifest = dict(schema="three_scene_evalset_v1", count_per_scene=100, total_episodes=300,
                episode_storage="byte-identical copied JSONs; external scene assets are symlinks",
                scenes=scenes, records=records,
                revised_source_manifest_sha256=hashlib.sha256((a.revised/"manifest.json").read_bytes()).hexdigest(),
                ours_checkpoint_sha256=h.hexdigest(),
                ours_selector_sha256=hashlib.sha256(selector.read_bytes()).hexdigest(),
                caveat="Native Full Warehouse is unchanged and not newly feasibility-filtered. Revised maps certify a 0.15m height slice only.")
(a.output/"manifest.json").write_text(json.dumps(manifest, indent=2))
(a.output/"README.txt").write_text(
    "Canonical three-scene evaluation set: 100 episodes per scene.\n"
    "Full_Warehouse: original native scripts, 15 pedestrians.\n"
    "Hospital / Office: quality100_p10_20261020 scripts, 10 pedestrians.\n"
    "Use scene_views/<Scene> as --scene_dir with --scene_index 0.\n"
    "Episode JSONs are local immutable snapshots, not links into older runs.\n"
    "Assets remain linked to the original DynBench installation.\n"
    "ours_reference_manifest.json specifies the exact GRPO u100 checkpoint and 16-candidate rule selector.\n"
    "Use the repaired shared evaluator/guard for NEW runs, not the reference run's obsolete guard.\n"
    "Old native Full Warehouse baseline numbers used the older guard and require protocol alignment for a strict comparison.\n")
print(json.dumps(dict(output=str(a.output), scenes=scenes), indent=2))

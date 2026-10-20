"""Gate actual scripting-loaded actor classes until the robot episode is ready."""

import json
import numpy as np
from omni.kit.scripting.scripts.script_manager import ScriptManager

holding = True
original_foreach = None


def install():
    global original_foreach
    original_foreach = ScriptManager._foreach_script_instance

    def foreach(manager, callback):
        def visit(inst):
            cls = type(inst)
            if hasattr(cls, "init_character") and not getattr(
                cls, "_episode_clock_installed", False
            ):
                original = cls.on_update

                def update(self, current_time, delta_time):
                    if holding:
                        if self.character is None:
                            if not self.init_character():
                                return
                            self.register_to_agent_manager()
                        self.character.set_variable("Action", "None")
                        if self.navigation_manager and self.avoidanceOn:
                            self.navigation_manager.publish_character_positions(
                                0.0, 0.5
                            )
                        return
                    return original(self, current_time, delta_time)

                cls.on_update = update
                cls._episode_clock_installed = True
            callback(inst)

        return original_foreach(manager, visit)

    ScriptManager._foreach_script_instance = foreach
    create = ScriptManager._create_script_instance

    def create_gated(manager, prim_path, script_path):
        inst = create(manager, prim_path, script_path)
        # New actors can be created while the timeline is already playing;
        # on_play traversal then never sees them before their first update.
        foreach(manager, lambda value: None)
        return inst

    ScriptManager._create_script_instance = create_gated


def hold():
    global holding
    holding = True


def release(episode_file):
    global holding
    with open(episode_file) as handle:
        expected = json.load(handle)["episode"]["characters"]["spawn_positions"]
    found = {}

    def inspect(inst):
        if not hasattr(inst, "character") or inst.character is None:
            return
        name = inst.get_agent_name()
        if name not in expected:
            return
        xy = np.asarray(inst.get_current_position())[:2]
        target = np.asarray(expected[name]["pos"])[:2]
        error = float(np.linalg.norm(xy - target))
        found[name] = dict(xy=xy.tolist(), error_m=error, commands=len(inst.commands))

    ScriptManager.get_instance()._foreach_script_instance(inspect)
    print("ACTOR_START " + json.dumps(found), flush=True)
    if set(found) != set(expected) or any(
        x["error_m"] > 0.15 or not x["commands"] for x in found.values()
    ):
        raise RuntimeError("Actor start positions/commands failed validation")
    holding = False

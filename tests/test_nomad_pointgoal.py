import importlib.util
from pathlib import Path
import sys
import types
import torch

# Selection is pure; stub optional model dependencies for a CPU unit test.
sys.modules.setdefault('base_agent', types.SimpleNamespace(NoMadBaseAgent=object))
sys.modules.setdefault('nomad_model', types.SimpleNamespace(NoMaDPolicy=object))
sys.modules.setdefault('traj_opt', types.SimpleNamespace(TrajOpt=object))
spec = importlib.util.spec_from_file_location(
    'nomad_agent', Path(__file__).parents[1]/'baselines/nomad/nomad_agent.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_pointgoal_endpoint_ranking_and_batch_independence():
    trajectories = torch.zeros(2, 3, 4, 2)
    trajectories[0, :, -1] = torch.tensor([[1., 0.], [2., 0.], [0., 2.]])
    trajectories[1, :, -1] = torch.tensor([[0., 1.], [0., -2.], [3., 0.]])
    chosen, costs, selected = module.select_pointgoal_trajectory(
        trajectories, torch.tensor([[1.8, 0.], [0., -1.8]]))
    assert selected.tolist() == [1, 1]
    assert torch.equal(chosen, trajectories[torch.arange(2), selected])
    assert torch.allclose(costs.min(dim=1).values, torch.tensor([.2, .2]), atol=1e-6)

import pytest
from episode_schedule import resolve_episode_ids


def test_sparse_ids_keep_original_numbers():
    assert resolve_episode_ids(0, 100, '1,2,7,18,97') == [1, 2, 7, 18, 97]
    assert resolve_episode_ids(7, 3) == [7, 8, 9]


@pytest.mark.parametrize('value', ['7,1', '1,1', '-1,2', ''])
def test_invalid_schedules(value):
    with pytest.raises(ValueError):
        resolve_episode_ids(0, 100, value)

"""Explicit episode IDs preserve scene reuse for sparse regression/recovery."""


def resolve_episode_ids(start, count, explicit=None):
    if explicit is None:
        if start < 0 or count <= 0:
            raise ValueError('Invalid episode range')
        return list(range(start, start + count))
    ids = [int(x) for x in explicit.split(',')]
    if not ids or min(ids) < 0 or ids != sorted(set(ids)):
        raise ValueError('episode_ids must be nonnegative, unique and increasing')
    return ids

from typing import List, Any
import numpy as np


def smart_collate(batch: List[Any], axis = 0) -> Any:
    """
    Recursively collates a batch of data from a vectorized environment into batches.

    Args:
        batch (List[Any]): List of elements for collating into a batch.

    Returns:
        Collated version where lists of elements are converted into batched structures.
    """
    elem = batch[0]

    # Handle a list of dicts
    if isinstance(elem, dict):
        return {key: smart_collate([d[key] for d in batch]) for key in elem}

    # Handle a list of tuples
    elif isinstance(elem, tuple):
        return tuple(smart_collate([d[i] for d in batch]) for i in range(len(elem)))

    # Handle a list of lists (for example, a list of environments' arrays)
    elif isinstance(elem, list):
        return smart_collate([smart_collate(sub_batch) for sub_batch in batch])

    # Handle a list of scalars or arrays
    else:
        # If everything is a numpy array or scalar, use np.stack
        return np.stack(batch, axis = axis)


def smart_concat(batch: List[Any], axis: int =0) -> Any:
    """
    Recursively collates a batch of data from a vectorized environment into batches.

    Args:
        batch (List[Any]): List of elements for collating into a batch.

    Returns:
        Collated version where lists of elements are converted into batched structures.
    """
    elem = batch[0]

    # Handle a list of dicts
    if isinstance(elem, dict):
        return {key: smart_concat([d[key] for d in batch]) for key in elem}

    # Handle a list of tuples
    elif isinstance(elem, tuple):
        return tuple(smart_concat([d[i] for d in batch]) for i in range(len(elem)))

    # Handle a list of lists (for example, a list of environments' arrays)
    elif isinstance(elem, list):
        return smart_concat([smart_concat(sub_batch) for sub_batch in batch])

    # Handle a list of scalars or arrays
    else:
        # If everything is a numpy array or scalar, use np.stack
        return np.concatenate(batch, axis=axis)
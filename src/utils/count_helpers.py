import numpy as np
from typing import Tuple


def get_sign(
    object_center: Tuple[float, float],
    line_start: Tuple[int, int],
    line_end: Tuple[int, int],
) -> int:
    """Determine the position of an object relative to a line."""

    d = (line_end[0] - line_start[0]) * (object_center[1] - line_start[1]) - (
        line_end[1] - line_start[1]
    ) * (object_center[0] - line_start[0])

    return np.sign(d)

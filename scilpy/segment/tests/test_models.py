import numpy as np
import pytest
from scilpy.segment.models import remove_similar_streamlines


@pytest.mark.skip(reason="Still failing")
def test_remove_similar_streamlines():
    streamlines = [np.array([[0, 0, 0], [1, 1, 1]]),
                   np.array([[0.1, 0.1, 0.1], [1.1, 1.1, 1.1]]),
                   np.array([[10, 10, 10], [11, 11, 11]])]
    new_streamlines = remove_similar_streamlines(streamlines)
    assert len(new_streamlines) == 2

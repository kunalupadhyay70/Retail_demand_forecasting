import pytest

from app.utils.common import chunk_list, utc_filename_timestamp, utc_timestamp


def test_timestamp_formats_and_chunking() -> None:
    assert len(utc_timestamp()) == 19
    assert utc_filename_timestamp().endswith("Z")
    assert list(chunk_list([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]

    with pytest.raises(ValueError, match="greater than 0"):
        list(chunk_list([1], 0))

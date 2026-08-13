from ludb_worst50_analysis import _rank_tail_status


def test_rank_tail_status_uses_worst_50_boundary() -> None:
    assert _rank_tail_status("150") == "outside"
    assert _rank_tail_status("151") == "inside"


def test_rank_tail_status_handles_missing_rank() -> None:
    assert _rank_tail_status("NA") == "unknown relative to"

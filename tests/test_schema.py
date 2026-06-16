from src.summarizer import SummaryResult


def test_summary_result_schema():
    result = SummaryResult(
        resume="Résumé de test.",
        points_cles=["Point 1", "Point 2"],
        actions=["Action 1"],
    )

    assert result.resume == "Résumé de test."
    assert len(result.points_cles) == 2
    assert result.actions[0] == "Action 1"


def test_summary_result_to_dict():
    result = SummaryResult(
        resume="Résumé.",
        points_cles=["A", "B"],
        actions=["C"],
    )

    data = result.model_dump()

    assert "resume" in data
    assert "points_cles" in data
    assert "actions" in data
    assert isinstance(data["points_cles"], list)
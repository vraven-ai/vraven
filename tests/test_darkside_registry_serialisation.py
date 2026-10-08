from __future__ import annotations

from pathlib import Path

from vraven.darkside import AnalysisResult, __version__, load_result, recommend_features


def test_recommendations_are_focused_by_community() -> None:
    security = recommend_features("security")
    training = recommend_features("training")
    assert security
    assert training
    assert security[0].name != training[0].name



def test_extension_version_is_exposed() -> None:
    assert __version__ == "0.10.1"

def test_result_round_trip(tmp_path: Path) -> None:
    result = AnalysisResult(
        "test",
        metrics={"value": 1.5},
        tables={"rows": [{"x": 1}]},
        limitations=["bounded test"],
    )
    path = result.save_json(tmp_path / "result.json")
    restored = load_result(path)
    assert restored.to_dict() == result.to_dict()

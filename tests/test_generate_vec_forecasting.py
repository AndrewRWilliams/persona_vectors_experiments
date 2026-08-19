import pandas as pd

from generate_vec import prepare_forecasting_persona_inputs


def test_prepare_forecasting_persona_inputs_filters_by_score(tmp_path):
    pos_path = tmp_path / "pos.csv"
    neg_path = tmp_path / "neg.csv"

    pd.DataFrame(
        [
            {"prompt": "p1", "answer": "a1", "forecasting_score": 0.8},
            {"prompt": "p2", "answer": "a2", "forecasting_score": 0.2},
        ]
    ).to_csv(pos_path, index=False)

    pd.DataFrame(
        [
            {"prompt": "q1", "answer": "b1", "forecasting_score": 0.9},
            {"prompt": "q2", "answer": "b2", "forecasting_score": 0.1},
        ]
    ).to_csv(neg_path, index=False)

    pos_prompts, neg_prompts, pos_responses, neg_responses = (
        prepare_forecasting_persona_inputs(
            str(pos_path),
            str(neg_path),
            score_column="forecasting_score",
            score_threshold=0.5,
        )
    )

    assert pos_prompts == ["p1"]
    assert neg_prompts == ["q2"]
    assert pos_responses == ["a1"]
    assert neg_responses == ["b2"]

from src.evaluator_blog import RougeScore
from rouge_score.scoring import Score

candidate = "it is dog"
reference = "it's dog"


def test_rouge_score():
    rg = RougeScore(
        candidates=candidate,
        references=reference,
    )
    assert rg.get_score() == {
        "rouge1": Score(precision=1.0, recall=1.0, fmeasure=1.0),
        "rouge2": Score(precision=0.0, recall=0.0, fmeasure=0.0),
        "rougeL": Score(precision=1.0, recall=1.0, fmeasure=1.0),
        "rougeLsum": Score(precision=1.0, recall=1.0, fmeasure=1.0),
    }

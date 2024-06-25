"""
Note: If you get ``TypeError: Fraction.__new__() got an unexpected keyword argument '_normalize'``
Then follow one of the following steps mentioned here: https://github.com/nltk/nltk/issues/3250#issuecomment-2132159186

P.S.: For me method-2 worked
"""

from src.evaluator_blog import BLEUScore

candidate = "it is dog".split()
references = [
    "it is dog".split(),
    "dog it is".split(),
    "a dog, it is".split(),
]

def test_bleu_score():
    bleu = BLEUScore(candidates=candidate, references=references, weights=(0, 0.5, 0.5, 0))
    assert bleu.get_score() == 0.5773502691896257

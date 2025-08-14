
def test_ngram_basic():
    from app_utils import NGramModel
    data = ["hello world","hello there","hello world test"]
    m = NGramModel(3)
    m.train_lines(data)
    out = m.predict("hello")
    assert isinstance(out, str)

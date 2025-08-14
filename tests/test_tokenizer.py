
def test_simple_tokenizer():
    from app_utils import SimpleTokenizer
    t = SimpleTokenizer()
    s = "Hello, world! This is a test."
    toks = t.tokenize(s)
    assert isinstance(toks, list)
    assert "Hello" in "".join(toks) or len(toks)>0

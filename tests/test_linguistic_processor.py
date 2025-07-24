from Python.deepseek_ja_adapter import JapaneseLinguisticProcessor


def test_variant_generation():
    proc = JapaneseLinguisticProcessor()
    text = "今日は良い天気です"
    variants = proc.generate_linguistic_variants(text, num_variants=2)
    assert text in variants
    assert isinstance(variants, list)
    assert len(variants) >= 1

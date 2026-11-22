import pytest

from linguabridge.translator import MarianTranslator


@pytest.fixture(scope="module")
def translator():
    return MarianTranslator()


def test_translates_with_pretrained_weights(translator):
    result = translator.translate("Hello, how are you?")
    assert result.lower().startswith("bonjour")
    assert result.endswith("?")


def test_rejects_blank_text(translator):
    with pytest.raises(ValueError, match="non-empty"):
        translator.translate(" \t")


def test_rejects_oversized_text(translator):
    limit = translator.model.config.max_position_embeddings
    with pytest.raises(ValueError, match="token limit"):
        translator.translate("hello " * (limit + 1))

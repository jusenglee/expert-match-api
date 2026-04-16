import sys
import types
from pathlib import Path

import pytest

from apps.search.encoders import (
    LocalSentenceTransformerEncoder,
    OpenAIEmbeddingEncoder,
    SpladeSparseEncoder,
)


def test_local_sentence_transformer_encoder_initializes_model_in_slots(monkeypatch):
    class FakeVector:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return list(self._values)

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            self.model_name = model_name
            self.kwargs = kwargs

        def encode(self, text):
            assert text == "sample query"
            return FakeVector([0.1, 0.2, 0.3])

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    encoder = LocalSentenceTransformerEncoder(model_name="local-model", vector_size=3)

    assert encoder.embed("sample query") == [0.1, 0.2, 0.3]


def test_local_sentence_transformer_encoder_uses_local_files_only_for_local_bundle(monkeypatch):
    class FakeVector:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return list(self._values)

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            self.model_name = model_name
            self.kwargs = kwargs

        def encode(self, text):
            assert text == "sample query"
            return FakeVector([0.1, 0.2, 0.3])

    model_dir = Path(__file__).resolve().parents[1] / "multilingual-e5-large-instruct"

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    encoder = LocalSentenceTransformerEncoder(model_name=str(model_dir), vector_size=3)

    assert encoder._model.model_name == str(model_dir.resolve())
    assert encoder._model.kwargs["local_files_only"] is True
    assert encoder.embed("sample query") == [0.1, 0.2, 0.3]


def test_local_sentence_transformer_encoder_rejects_incomplete_local_bundle(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("SentenceTransformer should not be initialized for an incomplete local bundle")

    model_dir = Path(__file__).resolve().parent / "fixtures" / "incomplete_st_bundle"

    fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    with pytest.raises(FileNotFoundError, match="1_Pooling"):
        LocalSentenceTransformerEncoder(model_name=str(model_dir), vector_size=3)


def test_splade_sparse_encoder_uses_local_files_only_for_local_bundle(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.device = None
            self.eval_called = False

        def to(self, device):
            self.device = device

        def eval(self):
            self.eval_called = True

    calls = {}
    model_dir = Path(__file__).resolve().parents[1] / "models" / "PIXIE-Splade-v1.0"

    def fake_tokenizer_from_pretrained(model_name, **kwargs):
        calls["tokenizer"] = (model_name, kwargs)
        return object()

    def fake_model_from_pretrained(model_name, **kwargs):
        calls["model"] = (model_name, kwargs)
        return FakeModel()

    monkeypatch.setattr(
        "apps.search.encoders.AutoTokenizer.from_pretrained",
        fake_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        "apps.search.encoders.AutoModelForMaskedLM.from_pretrained",
        fake_model_from_pretrained,
    )
    monkeypatch.setattr("apps.search.encoders.torch.cuda.is_available", lambda: False)

    encoder = SpladeSparseEncoder(model_name=str(model_dir))

    assert encoder.model_name == str(model_dir.resolve())
    assert calls["tokenizer"] == (
        str(model_dir.resolve()),
        {"local_files_only": True},
    )
    assert calls["model"][0] == str(model_dir.resolve())
    assert calls["model"][1] == {"local_files_only": True}
    assert encoder._model.device == "cpu"
    assert encoder._model.eval_called is True


def test_splade_sparse_encoder_rejects_missing_local_path(monkeypatch):
    missing_model_dir = Path(__file__).resolve().parent / "fixtures" / "missing_splade_bundle"

    monkeypatch.setattr(
        "apps.search.encoders.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("AutoTokenizer should not be called for a missing local path")
        ),
    )
    monkeypatch.setattr(
        "apps.search.encoders.AutoModelForMaskedLM.from_pretrained",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("AutoModel should not be called for a missing local path")
        ),
    )

    with pytest.raises(FileNotFoundError, match="missing_splade_bundle"):
        SpladeSparseEncoder(model_name=str(missing_model_dir))


def test_splade_sparse_encoder_passes_repo_id_with_local_files_only_override(monkeypatch):
    class FakeModel:
        def to(self, device):
            self.device = device

        def eval(self):
            self.eval_called = True

    calls = {}

    def fake_tokenizer_from_pretrained(model_name, **kwargs):
        calls["tokenizer"] = (model_name, kwargs)
        return object()

    def fake_model_from_pretrained(model_name, **kwargs):
        calls["model"] = (model_name, kwargs)
        return FakeModel()

    monkeypatch.setattr(
        "apps.search.encoders.AutoTokenizer.from_pretrained",
        fake_tokenizer_from_pretrained,
    )
    monkeypatch.setattr(
        "apps.search.encoders.AutoModelForMaskedLM.from_pretrained",
        fake_model_from_pretrained,
    )
    monkeypatch.setattr("apps.search.encoders.torch.cuda.is_available", lambda: False)

    encoder = SpladeSparseEncoder(
        model_name="telepix/PIXIE-Splade-v1.0",
        local_files_only=True,
    )

    assert encoder.model_name == "telepix/PIXIE-Splade-v1.0"
    assert calls["tokenizer"] == (
        "telepix/PIXIE-Splade-v1.0",
        {"local_files_only": True},
    )
    assert calls["model"] == (
        "telepix/PIXIE-Splade-v1.0",
        {"local_files_only": True},
    )


def test_openai_embedding_encoder_initializes_client_in_slots(monkeypatch):
    class FakeEmbeddingsApi:
        @staticmethod
        def create(*, model, input):
            assert model == "embed-model"
            assert input == "sample query"
            return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[0.4, 0.5])])

    class FakeOpenAIClient:
        def __init__(self, *, base_url, api_key):
            assert base_url == "http://localhost:8011/v1"
            assert api_key == "EMPTY"
            self.embeddings = FakeEmbeddingsApi()

    monkeypatch.setattr("apps.search.encoders.OpenAI", FakeOpenAIClient)

    encoder = OpenAIEmbeddingEncoder(
        model_name="embed-model",
        vector_size=2,
        base_url="http://localhost:8011/v1",
        api_key="EMPTY",
    )

    assert encoder.embed("sample query") == [0.4, 0.5]

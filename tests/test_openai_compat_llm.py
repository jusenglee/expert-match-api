import langchain

from apps.core.openai_compat_llm import OpenAICompatChatModel, _ensure_langchain_global_compat


def test_langchain_global_compat_backfills_expected_root_attributes():
    for attr in ("verbose", "debug", "llm_cache"):
        if hasattr(langchain, attr):
            delattr(langchain, attr)

    _ensure_langchain_global_compat()

    assert langchain.verbose is False
    assert langchain.debug is False
    assert langchain.llm_cache is None


def test_openai_compat_chat_model_initializes_with_backfilled_langchain_globals():
    for attr in ("verbose", "debug", "llm_cache"):
        if hasattr(langchain, attr):
            delattr(langchain, attr)

    _ensure_langchain_global_compat()
    model = OpenAICompatChatModel(
        model_name="/model",
        base_url="http://127.0.0.1:8010/v1",
        api_key="EMPTY",
    )

    assert model.model_name == "/model"

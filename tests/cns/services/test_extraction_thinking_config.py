"""Regression tests for extraction thinking config wiring."""

from types import SimpleNamespace

from config.config import BatchingConfig, ExtractionConfig
from lt_memory.batching import BatchingService
from lt_memory.processing.execution_strategy import (
    BatchExecutionStrategy,
    ImmediateExecutionStrategy,
)

USER_ID = "00000000-0000-0000-0000-000000000001"


class _DummyBatchCreate:
    def __init__(self):
        self.requests = None

    def create(self, requests):
        self.requests = requests
        return SimpleNamespace(id="batch_test_1")


class _DummyAnthropicClient:
    def __init__(self):
        self.batch_create = _DummyBatchCreate()
        self.beta = SimpleNamespace(
            messages=SimpleNamespace(
                batches=self.batch_create
            )
        )


class _DummyDB:
    def __init__(self):
        self.created = []

    def create_extraction_batch(self, batch_record, user_id=None):
        self.created.append((batch_record, user_id))

    def update_extraction_timestamp(self, user_id):
        return user_id


class _DummyExtractionEngine:
    def __init__(self, extraction_config):
        self.config = extraction_config

    def build_extraction_payload(self, chunk, for_batch=True):
        if for_batch:
            return SimpleNamespace(
                messages=[{"role": "user", "content": "extract"}],
                system_prompt="system",
                short_to_uuid={},
                memory_context={},
            )
        return SimpleNamespace(
            user_prompt="extract now",
            system_prompt="system",
            short_to_uuid={},
            memory_context={},
        )


class _DummyLLMProvider:
    def __init__(self):
        self.calls = []

    def generate_response(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content="[]")

    def extract_text_content(self, response):
        return "[]"

    def _is_failover_active(self):
        return False


def _chunk(index=0):
    return SimpleNamespace(chunk_index=index, messages=[{"role": "user"}], memory_context_snapshot={})


def test_batch_execution_strategy_uses_extraction_thinking_config():
    extraction_config = ExtractionConfig(
        extraction_thinking_enabled=False,
        extraction_thinking_budget=777,
    )
    strategy = BatchExecutionStrategy(
        extraction_engine=_DummyExtractionEngine(extraction_config),
        memory_processor=SimpleNamespace(),
        vector_ops=SimpleNamespace(),
        db=_DummyDB(),
        anthropic_client=_DummyAnthropicClient(),
        batching_config=BatchingConfig(),
        extraction_config=extraction_config,
    )

    strategy.execute_extraction(USER_ID, [_chunk(1)])
    params = strategy.anthropic_client.batch_create.requests[0]["params"]
    assert "thinking" not in params


def test_batching_service_submit_uses_extraction_thinking_config():
    service = BatchingService.__new__(BatchingService)
    service.config = BatchingConfig()
    service.db = _DummyDB()
    service.extraction_prompt = "system"
    service.extraction = SimpleNamespace(
        config=ExtractionConfig(
            extraction_thinking_enabled=True,
            extraction_thinking_budget=555,
        ),
        build_extraction_payload=lambda chunk: {"short_to_uuid": {}},
    )
    service._build_batch_messages = lambda chunk: [{"role": "user", "content": "msg"}]
    service.llm_provider = _DummyLLMProvider()
    service.anthropic_client = _DummyAnthropicClient()

    service._submit_extraction_batch(USER_ID, [_chunk(2)])
    params = service.anthropic_client.batch_create.requests[0]["params"]
    assert params["thinking"]["type"] == "enabled"
    assert params["thinking"]["budget_tokens"] == 555


def test_immediate_execution_strategy_uses_extraction_thinking_config():
    extraction_config = ExtractionConfig(
        extraction_thinking_enabled=False,
        extraction_thinking_budget=222,
    )
    llm = _DummyLLMProvider()
    strategy = ImmediateExecutionStrategy(
        extraction_engine=_DummyExtractionEngine(extraction_config),
        memory_processor=SimpleNamespace(),
        vector_ops=SimpleNamespace(),
        db=_DummyDB(),
        llm_provider=llm,
    )
    strategy._process_and_store_memories = lambda user_id, response_text, payload: ([], [])

    strategy.execute_extraction(USER_ID, [_chunk(3)])
    assert llm.calls, "Expected at least one LLM call"
    assert llm.calls[0]["thinking_enabled"] is False
    assert llm.calls[0]["thinking_budget"] == 222

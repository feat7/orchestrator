"""Tests for intent classification."""

import pytest
from app.core.intent import IntentClassifier
from app.schemas.intent import ServiceType, StepType


@pytest.mark.asyncio
async def test_intent_classifier_initialization(mock_llm):
    """Test IntentClassifier initialization."""
    classifier = IntentClassifier(mock_llm)
    assert classifier.llm == mock_llm


@pytest.mark.asyncio
async def test_classify_single_service(mock_llm):
    """Test classification of single-service query."""
    classifier = IntentClassifier(mock_llm)
    intent = await classifier.classify("What's on my calendar next week?")

    assert len(intent.services) > 0
    assert len(intent.steps) > 0
    assert intent.confidence > 0


@pytest.mark.asyncio
async def test_classify_returns_valid_services(mock_llm):
    """Test that classification returns valid service types."""
    classifier = IntentClassifier(mock_llm)
    intent = await classifier.classify("Find emails from john@example.com")

    # Services are strings due to use_enum_values=True in ParsedIntent
    valid_services = [s.value for s in ServiceType]
    for service in intent.services:
        assert service in valid_services


@pytest.mark.asyncio
async def test_classify_returns_valid_steps(mock_llm):
    """Test that classification returns valid step types."""
    classifier = IntentClassifier(mock_llm)
    intent = await classifier.classify("Search my drive for budget documents")

    # Steps are strings due to use_enum_values=True in ParsedIntent
    valid_steps = [s.value for s in StepType]
    for step in intent.steps:
        assert step in valid_steps


@pytest.mark.asyncio
async def test_classify_with_context(mock_llm):
    """Test classification with conversation context."""
    classifier = IntentClassifier(mock_llm)
    context = [{"query": "Find emails from sarah@company.com"}]

    intent = await classifier.classify("Show me the latest one", context)

    assert intent is not None
    assert len(intent.services) > 0


def test_parse_json_response():
    """Test JSON parsing from various formats."""
    from app.core.intent import IntentClassifier

    # Create classifier with mock (doesn't matter for this test)
    classifier = IntentClassifier(None)

    # Test direct JSON
    result = classifier._parse_json_response('{"services": ["gmail"]}')
    assert result["services"] == ["gmail"]

    # Test markdown code block
    result = classifier._parse_json_response('```json\n{"services": ["gcal"]}\n```')
    assert result["services"] == ["gcal"]


def test_build_intent_with_invalid_services():
    """Test that invalid services are skipped."""
    from app.core.intent import IntentClassifier

    classifier = IntentClassifier(None)

    data = {
        "services": ["gmail", "invalid_service", "gcal"],
        "operation": "search",
        "steps": ["search_gmail"],
    }

    intent = classifier._build_intent(data)

    # Services are strings due to use_enum_values=True in ParsedIntent
    # Should have gmail and gcal, but not invalid_service
    assert "gmail" in intent.services
    assert "gcal" in intent.services
    assert "invalid_service" not in intent.services
    assert len(intent.services) == 2


def test_build_intent_defaults():
    """Test default values when data is incomplete."""
    from app.core.intent import IntentClassifier

    classifier = IntentClassifier(None)

    # Empty data should return defaults
    intent = classifier._build_intent({})

    assert len(intent.services) > 0  # Should have default service
    assert len(intent.steps) > 0  # Should have default step
    assert intent.operation == "search"  # Default operation

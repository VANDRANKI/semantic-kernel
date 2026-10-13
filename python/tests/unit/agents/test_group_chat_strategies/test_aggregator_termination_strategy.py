# Copyright (c) Microsoft. All rights reserved.

from unittest.mock import AsyncMock, MagicMock

import pytest

from semantic_kernel.agents.group_chat.agent_group_chat import AgentGroupChat
from semantic_kernel.agents.strategies.termination.aggregator_termination_strategy import (
    AggregateTerminationCondition,
    AggregatorTerminationStrategy,
)
from semantic_kernel.agents.strategies.termination.termination_strategy import TerminationStrategy
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from tests.unit.agents.test_agent import MockAgent


async def test_aggregate_termination_condition_all_true():
    agent = MockAgent(id="test-agent-id")
    history = [MagicMock(spec=ChatMessageContent)]

    # Mocking two strategies that return True
    strategy1 = AsyncMock(spec=TerminationStrategy)
    strategy1.should_terminate.return_value = True

    strategy2 = AsyncMock(spec=TerminationStrategy)
    strategy2.should_terminate.return_value = True

    strategy = AggregatorTerminationStrategy(
        strategies=[strategy1, strategy2], condition=AggregateTerminationCondition.ALL
    )

    result = await strategy.should_terminate_async(agent, history)

    assert result is True
    strategy1.should_terminate.assert_awaited_once()
    strategy2.should_terminate.assert_awaited_once()


async def test_aggregate_termination_condition_all_false():
    agent = MockAgent(id="test-agent-id")
    history = [MagicMock(spec=ChatMessageContent)]

    # Mocking two strategies, one returns True, the other False
    strategy1 = AsyncMock(spec=TerminationStrategy)
    strategy1.should_terminate.return_value = True

    strategy2 = AsyncMock(spec=TerminationStrategy)
    strategy2.should_terminate.return_value = False

    strategy = AggregatorTerminationStrategy(
        strategies=[strategy1, strategy2], condition=AggregateTerminationCondition.ALL
    )

    result = await strategy.should_terminate_async(agent, history)

    assert result is False
    strategy1.should_terminate.assert_awaited_once()
    strategy2.should_terminate.assert_awaited_once()


async def test_aggregate_termination_condition_any_true():
    agent = MockAgent(id="test-agent-id")
    history = [MagicMock(spec=ChatMessageContent)]

    # Mocking two strategies, one returns False, the other True
    strategy1 = AsyncMock(spec=TerminationStrategy)
    strategy1.should_terminate.return_value = False

    strategy2 = AsyncMock(spec=TerminationStrategy)
    strategy2.should_terminate.return_value = True

    strategy = AggregatorTerminationStrategy(
        strategies=[strategy1, strategy2], condition=AggregateTerminationCondition.ANY
    )

    result = await strategy.should_terminate_async(agent, history)

    assert result is True
    strategy1.should_terminate.assert_awaited_once()
    strategy2.should_terminate.assert_awaited_once()


async def test_aggregate_termination_condition_any_false():
    agent = MockAgent(id="test-agent-id")
    history = [MagicMock(spec=ChatMessageContent)]

    # Mocking two strategies that return False
    strategy1 = AsyncMock(spec=TerminationStrategy)
    strategy1.should_terminate.return_value = False

    strategy2 = AsyncMock(spec=TerminationStrategy)
    strategy2.should_terminate.return_value = False

    strategy = AggregatorTerminationStrategy(
        strategies=[strategy1, strategy2], condition=AggregateTerminationCondition.ANY
    )

    result = await strategy.should_terminate_async(agent, history)

    assert result is False
    strategy1.should_terminate.assert_awaited_once()
    strategy2.should_terminate.assert_awaited_once()


def test_aggregator_is_a_termination_strategy():
    strategy = AggregatorTerminationStrategy(strategies=[AsyncMock(spec=TerminationStrategy)])

    assert isinstance(strategy, TerminationStrategy)
    assert AgentGroupChat(termination_strategy=strategy).termination_strategy is strategy


@pytest.mark.parametrize(
    ("condition", "results", "expected"),
    [
        (AggregateTerminationCondition.ALL, [True, True], True),
        (AggregateTerminationCondition.ALL, [True, False], False),
        (AggregateTerminationCondition.ANY, [False, True], True),
        (AggregateTerminationCondition.ANY, [False, False], False),
    ],
)
async def test_aggregator_should_terminate(condition, results, expected):
    agent = MockAgent(id="test-agent-id")
    history = [MagicMock(spec=ChatMessageContent)]

    children = []
    for result in results:
        child = AsyncMock(spec=TerminationStrategy)
        child.should_terminate.return_value = result
        children.append(child)

    strategy = AggregatorTerminationStrategy(strategies=children, condition=condition)

    assert await strategy.should_terminate(agent, history) is expected

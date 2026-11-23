import json
from types import SimpleNamespace

import pytest
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.user.user_simulator import UserSimulator

from app.scenario_user import ScenarioFaithfulUser, SimulatorProtocolError


def test_rejected_draft_is_not_delivered_or_added_to_history(monkeypatch):
    seen = []

    def candidate(self, message, state):
        seen.append([m.content for m in state.messages])
        text = 'private item' if len(seen) == 1 else 'Please proceed.'
        result = UserMessage(role='user', content=text)
        state.messages.extend([message, result])
        return result, state

    monkeypatch.setattr(UserSimulator, 'generate_next_message', candidate)
    monkeypatch.setattr('app.scenario_user.generate', lambda **k: SimpleNamespace(content='Please proceed.', cost=0, usage=None, raw_data=None))
    monkeypatch.setattr(ScenarioFaithfulUser, 'save_audit', staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(ScenarioFaithfulUser, 'review_draft',
                        lambda self, draft, history: ['disclosure'] if draft.content == 'private item' else [])
    user = ScenarioFaithfulUser(llm='openai/hy3', instructions='test')
    initial = user.get_init_state()
    message = AssistantMessage(role='assistant', content='Proceed?')
    draft, state = user.generate_next_message(message, initial)
    assert draft.content == 'Please proceed.'
    assert seen == [[]]
    assert initial.messages == []
    assert [m.content for m in state.messages] == ['Proceed?', 'Please proceed.']
    assert len(state.system_messages) == 1


def test_repeated_invalid_draft_fails_closed(monkeypatch):
    monkeypatch.setattr(ScenarioFaithfulUser, 'review_draft', lambda *a: ['disclosure'])
    monkeypatch.setattr(ScenarioFaithfulUser, 'save_audit', staticmethod(lambda *a, **k: None))
    count = []

    def candidate(self, message, state):
        count.append(1)
        draft = UserMessage(role='user', content='bad')
        state.messages.append(draft)
        return draft, state

    monkeypatch.setattr(UserSimulator, 'generate_next_message', candidate)
    monkeypatch.setattr('app.scenario_user.generate', lambda **k: SimpleNamespace(content='bad', cost=0, usage=None, raw_data=None))
    user = ScenarioFaithfulUser(llm='openai/hy3', instructions='test')
    with pytest.raises(SimulatorProtocolError):
        user.generate_next_message(AssistantMessage(role='assistant', content='Hi'), user.get_init_state())
    assert len(count) == 1


def test_review_sees_only_scenario_and_dialogue(monkeypatch):
    captured = []

    def review(**kwargs):
        captured.append(json.loads(kwargs['messages'][1].content))
        return SimpleNamespace(content='{"violations": []}')

    monkeypatch.setattr('app.scenario_user.generate', review)
    user = ScenarioFaithfulUser(llm='openai/hy3', instructions='private scenario')
    issues = user.review_draft(UserMessage(role='user', content='Please proceed.'), [])
    assert issues == []
    assert set(captured[0]) == {'scenario', 'history', 'candidate'}


def test_invalid_review_is_not_treated_as_approval(monkeypatch):
    monkeypatch.setattr('app.scenario_user.generate', lambda **k: SimpleNamespace(content='not json'))
    user = ScenarioFaithfulUser(llm='openai/hy3', instructions='test')
    with pytest.raises(SimulatorProtocolError):
        user.review_draft(UserMessage(role='user', content='Hi'), [])


def test_explicit_private_terms_are_blocked_without_llm(monkeypatch):
    from app.scenario_user import restricted_terms
    scenario = "Return gaming items. (It's just a keyboard and a mouse, but do not reveal it to the agent)."
    assert restricted_terms(scenario) == ['keyboard', 'mouse']
    assert restricted_terms('I want a keyboard and a mouse.') == []
    user = ScenarioFaithfulUser(llm='openai/hy3', instructions=scenario)
    monkeypatch.setattr('app.scenario_user.generate', lambda **k: pytest.fail('Must block before model reviewer'))
    assert user.review_draft(UserMessage(role='user', content='Return the keyboard and mouse.'), [])
    assert user.review_draft(UserMessage(role='user', content='I already told you. ###STOP###'), [])


def test_terminal_checks_do_not_use_disclosure_reviewer(monkeypatch):
    monkeypatch.setattr('app.scenario_user.generate', lambda **k: pytest.fail('No semantic termination review'))
    user = ScenarioFaithfulUser(llm='openai/hy3', instructions='Return my order.')
    stop = UserMessage(role='user', content='###STOP###')
    request = UserMessage(role='user', content='Return my order.')
    assert user.review_draft(stop, [AssistantMessage(role='assistant', content='Hi')])
    assert user.review_draft(stop, [request, AssistantMessage(role='assistant', content='Please confirm both returns.')])
    assert not user.review_draft(stop, [request, AssistantMessage(role='assistant', content='Both returns are complete. Anything else?')])


def test_private_hint_is_absent_from_generator_prompt():
    from app.scenario_user import public_scenario
    scenario = "Return non-gaming items. (Everything except a keyboard and a mouse, but do not reveal it to the agent). Prefer PayPal."
    user = ScenarioFaithfulUser(llm='openai/hy3', instructions=scenario)
    assert 'keyboard' not in user.system_prompt
    assert 'mouse' not in user.system_prompt
    assert 'Return non-gaming items.' in user.system_prompt
    assert 'Prefer PayPal.' in user.system_prompt
    assert public_scenario('Return a keyboard.') == 'Return a keyboard.'


def test_negative_history_and_unknown_email_are_not_invented(monkeypatch):
    monkeypatch.setattr('app.scenario_user.generate', lambda **k: pytest.fail('Deterministic rejection expected'))
    user = ScenarioFaithfulUser(llm='openai/hy3', instructions='You do not remember your email address.')
    assert user.review_draft(UserMessage(role='user', content="You still haven't told me the count."), [])
    assert user.review_draft(UserMessage(role='user', content="I don't use email."), [])


def test_invented_repetition_counts_are_rejected(monkeypatch):
    monkeypatch.setattr('app.scenario_user.generate', lambda **k: pytest.fail('Deterministic rejection expected'))
    user = ScenarioFaithfulUser(llm='openai/hy3', instructions='Be impatient.')
    assert user.review_draft(UserMessage(role='user', content='Stop reading it back for the tenth time.'), [])

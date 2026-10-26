"""JSON recovery and configuration migration, with no real model requests."""
import copy
import json

import httpx
import pytest

from benchmarks.amem_adapter import CompletionAdapter, AMemProviderStop, JSON_RECOVERY_POLICY
from benchmarks.comparison_worker import CachedClients, atomic_json
from benchmarks.findingdory_structured import fingerprint
from benchmarks.run_budget import RunBudget
from benchmarks import run_validation50_amem as runner
from memory_agent.settings import Settings

FORMAT = {'type': 'json_schema', 'json_schema': {'schema': {
    'type': 'object', 'properties': {'should_evolve': {'type': 'boolean'}},
    'required': ['should_evolve'], 'additionalProperties': False}}}


def response(content, reason='stop'):
    return {'model': 'mock', 'choices': [{'finish_reason': reason, 'message': {'content': content}}],
            'usage': {'total_tokens': 10}}


def test_cached_truncation_is_recovered_once_and_validated_cache_is_reused(tmp_path):
    sent = []

    def transport(request):
        payload = json.loads(request.content)
        sent.append(payload)
        return httpx.Response(200, json=(response('{"should_evolve":', 'length')
            if payload['max_tokens'] == 4096 else response('{"should_evolve": false}')))

    settings = Settings(llm_model='mock', llm_key='fake')
    budget = RunBudget(tmp_path, max_requests=10, max_reported_tokens=0)
    client = CachedClients(settings, budget, tmp_path/'response_cache', transport=httpx.MockTransport(transport))
    adapter = CompletionAdapter(client, tmp_path/'memory')
    # Recreate the old failure: HTTP response was paid and cached before JSON parsing.
    original = {'model': 'mock', 'temperature': 1.0, 'max_tokens': 4096,
                'response_format': {'type': 'json_object'}, **client.chat_options('mock'),
                'messages': [{'role': 'system', 'content': 'You must respond with a JSON object.'},
                             {'role': 'user', 'content': 'original memory evolution prompt'}]}
    client.post(settings.llm_url.rstrip('/')+'/chat/completions', 'fake', original, 'amem_evolution')
    assert budget.state['requests_started'] == 1
    try:
        assert json.loads(adapter.get_completion('original memory evolution prompt', FORMAT)) == {'should_evolve': False}
        assert [p['max_tokens'] for p in sent] == [4096, 8192]
        assert sent[0]['messages'] == sent[1]['messages']
        assert budget.state['requests_started'] == 2
        assert budget.state['reported_tokens'] == 20
        assert len(list((tmp_path/'memory/json_recovery').glob('*/round_0_attempt_0.json'))) == 1
        # A restarted adapter consumes its accepted cache, with no HTTP attempt.
        assert json.loads(CompletionAdapter(client, tmp_path/'memory').get_completion(
            'original memory evolution prompt', FORMAT)) == {'should_evolve': False}
        assert len(sent) == 2
    finally:
        client.close()


class FakeClient:
    settings = Settings(llm_model='mock', llm_key='fake')

    def __init__(self, responses):
        self.responses = iter(responses)
        self.payloads = []

    def chat_options(self, model):
        return {}

    def post(self, url, key, payload, role):
        self.payloads.append(copy.deepcopy(payload))
        return next(self.responses)


def test_truncation_recovery_is_bounded_and_never_fabricates_memory(tmp_path):
    client = FakeClient([response('{"should_evolve":', 'length') for _ in range(3)])
    with pytest.raises(AMemProviderStop, match='bounded recovery'):
        CompletionAdapter(client, tmp_path).get_completion('memory only', FORMAT)
    assert [p['max_tokens'] for p in client.payloads] == [4096, 8192, 16384]
    assert not list((tmp_path/'completions').glob('*.json'))
    assert len(list((tmp_path/'json_recovery').glob('*/round_0_attempt_*.json'))) == 3


def test_manual_resume_after_exhausted_retries_can_request_a_fresh_repair(tmp_path):
    sent = []

    def transport(request):
        payload = json.loads(request.content)
        sent.append(payload)
        complete = any('JSON recovery cycle 2' in msg['content'] for msg in payload['messages'])
        return httpx.Response(200, json=response('{"should_evolve": false}' if complete else '{'))

    budget = RunBudget(tmp_path, max_requests=10, max_reported_tokens=0)
    client = CachedClients(Settings(llm_model='mock', llm_key='fake'), budget,
                           tmp_path/'response_cache', transport=httpx.MockTransport(transport))
    adapter = CompletionAdapter(client, tmp_path/'memory')
    try:
        with pytest.raises(AMemProviderStop):
            adapter.get_completion('memory', FORMAT)
        assert len(sent) == 3
        # Original bad response is reused for diagnosis; only one fresh repair is billed.
        assert json.loads(adapter.get_completion('memory', FORMAT)) == {'should_evolve': False}
        assert len(sent) == 4
    finally:
        client.close()


def test_schema_failure_is_retried_using_original_schema(tmp_path):
    client = FakeClient([response('{}'), response('{"should_evolve": true}')])
    result = CompletionAdapter(client, tmp_path).get_completion('original memory', FORMAT)
    assert json.loads(result) == {'should_evolve': True}
    assert len(client.payloads) == 2
    assert client.payloads[1]['messages'][:2] == client.payloads[0]['messages']
    assert 'should_evolve' in client.payloads[1]['messages'][-1]['content']


def test_fenced_complete_json_is_unwrapped_without_another_request(tmp_path):
    client = FakeClient([response('```json\n{"should_evolve": false}\n```')])
    assert json.loads(CompletionAdapter(client, tmp_path).get_completion('memory', FORMAT)) == {'should_evolve': False}
    assert len(client.payloads) == 1


def test_invalid_accepted_cache_does_not_bypass_schema_validation(tmp_path):
    client = FakeClient([response('{"should_evolve": false}')])
    adapter = CompletionAdapter(client, tmp_path)
    adapter.get_completion('memory', FORMAT)
    accepted = next((tmp_path/'completions').glob('*.json'))
    atomic_json(accepted, {'content': '{}'})
    client.responses = iter([response('{"should_evolve": true}')])
    assert json.loads(adapter.get_completion('memory', FORMAT)) == {'should_evolve': True}
    assert list((tmp_path/'json_recovery').glob('*/invalid_accepted_cache.json'))


def configs():
    old = {'protocol': 'findingdory-validation50-amem-v1', 'models': {'llm_model': 'unchanged'},
           'source_sha256': {**runner.LEGACY_JSON_RECOVERY_SOURCES, 'shared.py': 'unchanged'},
           'tasks_sha256': 'same_tasks', 'observations_manifest_sha256': 'same_observations'}
    new = copy.deepcopy(old)
    new['amem_json_recovery'] = JSON_RECOVERY_POLICY
    for name in runner.LEGACY_JSON_RECOVERY_SOURCES:
        new['source_sha256'][name] = 'current_'+name
    return old, new


def test_exact_legacy_config_is_migrated_with_backup_and_audit(tmp_path):
    old, new = configs()
    frozen_old = old | {'fingerprint': fingerprint(old)}
    atomic_json(tmp_path/'config.json', frozen_old)
    atomic_json(tmp_path/'answers/12.json', {'unchanged': True})
    runner.freeze_run_config(tmp_path, new)
    assert json.loads((tmp_path/'compatibility/config_before_amem_json_recovery_v1.json').read_text()) == frozen_old
    expected = new | {'fingerprint': fingerprint(new)}
    assert json.loads((tmp_path/'config.json').read_text()) == expected
    audit_path = tmp_path/'compatibility/amem_json_recovery_v1.json'
    audit = audit_path.read_bytes()
    runner.freeze_run_config(tmp_path, new)
    assert audit_path.read_bytes() == audit
    assert json.loads((tmp_path/'answers/12.json').read_text()) == {'unchanged': True}
    with pytest.raises(RuntimeError, match='beyond'):
        runner.freeze_run_config(tmp_path, new | {'tasks_sha256': 'different'})


@pytest.mark.parametrize('change', ['model', 'unknown_source', 'fingerprint'])
def test_migration_rejects_other_changes(tmp_path, change):
    old, new = configs()
    if change == 'model':
        new['models']['llm_model'] = 'different'
    if change == 'unknown_source':
        old['source_sha256']['benchmarks/amem_adapter.py'] = 'unknown_version'
    prior = old | {'fingerprint': fingerprint(old)}
    if change == 'fingerprint':
        prior['fingerprint'] = 'tampered'
    atomic_json(tmp_path/'config.json', prior)
    with pytest.raises(RuntimeError, match='beyond'):
        runner.freeze_run_config(tmp_path, new)
    assert not (tmp_path/'compatibility').exists()
    assert json.loads((tmp_path/'config.json').read_text()) == prior


def test_migration_can_resume_if_interrupted_before_config_replacement(tmp_path, monkeypatch):
    old, new = configs()
    prior = old | {'fingerprint': fingerprint(old)}
    atomic_json(tmp_path/'config.json', prior)
    original_writer = runner.atomic_json

    def interrupted_writer(path, value):
        if path == tmp_path/'config.json':
            raise OSError('simulated interruption')
        original_writer(path, value)

    monkeypatch.setattr(runner, 'atomic_json', interrupted_writer)
    with pytest.raises(OSError, match='interruption'):
        runner.freeze_run_config(tmp_path, new)
    assert json.loads((tmp_path/'config.json').read_text()) == prior
    monkeypatch.setattr(runner, 'atomic_json', original_writer)
    runner.freeze_run_config(tmp_path, new)
    assert json.loads((tmp_path/'config.json').read_text())['fingerprint'] == fingerprint(new)

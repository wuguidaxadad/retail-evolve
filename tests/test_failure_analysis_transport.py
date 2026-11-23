from types import SimpleNamespace as NS

from app import failure_analysis as f


def test_empty_analysis_retries_and_requires_json_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(f, 'ROOT', tmp_path)
    sent = []
    def create(**kw):
        sent.append(kw)
        content = '' if len(sent) == 1 else '{"clusters":[],"summary":"All pass"}'
        return NS(choices=[NS(finish_reason='stop',message=NS(content=content))],
                  model_dump=lambda **kw: {'content':content})
    monkeypatch.setattr(f, 'OpenAI', lambda **kw: NS(chat=NS(completions=NS(create=create))))
    result = f.analyze_training([{'task_id':'1','messages':[], 'reward_info':{'reward':1}}], 'hash', ['1'])
    assert len(sent) == 2
    assert all(s['response_format'] == {'type':'json_object'} for s in sent)
    assert '<execution_evidence>' in sent[0]['messages'][1]['content']
    assert 'Now analyze' in sent[0]['messages'][1]['content']
    assert result['source_hash'] == 'hash'
    assert len(list((tmp_path/'.runtime/failure-analysis-responses').glob('*.json'))) == 2


def test_analysis_cannot_invent_failure_labels(monkeypatch, tmp_path):
    import json
    monkeypatch.setattr(f, 'ROOT', tmp_path)
    calls = []
    def create(**kw):
        calls.append(kw)
        cluster = {'id':'c1','failed_task_ids':['pass' if len(calls)==1 else 'fail'],
                   'control_task_ids':[], 'edit_surface':'state','root_cause':'lost selection',
                   'hypothesis':'track selection'}
        content = json.dumps({'clusters':[cluster]})
        return NS(choices=[NS(finish_reason='stop',message=NS(content=content))],
                  model_dump=lambda **kw: {'content':content})
    monkeypatch.setattr(f, 'OpenAI', lambda **kw: NS(chat=NS(completions=NS(create=create))))
    rows = [{'task_id':t,'messages':[], 'reward_info':{'reward':r}} for t,r in [('pass',1),('fail',0)]]
    result = f.analyze_training(rows, 'hash', ['pass','fail'])
    assert len(calls)==2
    assert result['clusters'][0]['failed_task_ids']==['fail']

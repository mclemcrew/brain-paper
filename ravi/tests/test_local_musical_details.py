import copy
from ravi.local.metadata import edit_track, crop_plan, conditioning_text, FORMAT_VERSION
from ravi.local.analysis import structured_suggestions


def track():
    return dict(name='Piece', family='piece', split='train', caption='Bowed strings',
                duration=8, sample_rate=44100, reviewed=True)


def detail(value, reviewed=True):
    return {'value': value, 'reviewed': reviewed, 'source': 'Composer'}


def test_only_confirmed_details_reach_training_and_old_text_is_unchanged():
    assert conditioning_text('My words', {}) == 'My words'
    assert conditioning_text('My words', {'instruments': detail('violin'),
        'tempo': detail('180 BPM', False)}) == 'My words\nInstruments: violin.'


def test_edit_invalidates_approval_and_history_preserves_details():
    t = track()
    edit_track(t, {'musical_details': {'texture': detail('sparse')}})
    assert not t['reviewed']
    edit_track(t, {'musical_details': {'texture': detail('dense')}})
    assert t['metadata_history'][-1]['musical_details']['texture']['value'] == 'sparse'
    edit_track(t, {'reviewed': True})
    assert 'Texture: dense.' in crop_plan(t, 4)['clips'][0]['caption']


def test_passage_details_do_not_inherit_global_instruments():
    t = track()
    edit_track(t, {'musical_details': {'instruments': detail('drums')}, 'sections': [
        {'start': 0, 'end': 4, 'caption': 'Quiet opening', 'musical_details': {'instruments': detail('violin')}},
        {'start': 4, 'end': 8, 'caption': 'Percussion enters'}], 'reviewed': True})
    clips = crop_plan(t, 3)['clips']
    assert all(c['end'] <= 4 if c['start'] < 4 else c['start'] >= 4 for c in clips)
    assert 'violin' in clips[0]['caption'] and 'drums' not in clips[0]['caption']
    assert clips[-1]['caption'] == 'Percussion enters'
    assert all(c['caption_format'] == FORMAT_VERSION for c in clips)


def test_automatic_evidence_keeps_scope_and_is_not_approved():
    tags = {'instrumentation': [{'label': 'piano', 'similarity': .25}]}
    draft = structured_suggestions(tags, 10, 20)
    original = copy.deepcopy(draft)
    assert conditioning_text('', draft) == ''
    t = track()
    edit_track(t, {'musical_details': draft})
    evidence = t['musical_details']['instruments']['suggestion']
    assert evidence['start'] == 10 and evidence['end'] == 20
    assert evidence['score_type'] == 'ranking_similarity'
    assert draft == original


def test_dismissed_suggestion_survives_save_without_training_text():
    t = track()
    edit_track(t, {'musical_details': {'instruments': {**detail(''), 'dismissed': True}}})
    assert t['musical_details']['instruments']['dismissed']
    assert conditioning_text(t['caption'], t['musical_details']) == t['caption']


def test_saved_details_match_prepared_audio_caption(tmp_path):
    import json
    import numpy as np
    import soundfile as sf
    from ravi.local.store import Store
    from ravi.local.worker import prepare
    from ravi.local.metadata import preview
    wav = tmp_path / 'source.wav'
    sf.write(wav, .1 * np.sin(np.arange(88200) * 2 * np.pi * 220 / 44100), 44100)
    store = Store(tmp_path / 'store')
    p = store.create('Metadata integration')
    p = store.import_file(p['id'], wav, wav.name)['project']
    t = p['tracks'][0]
    store.update(p['id'], {'track': {'id': t['id'], 'caption': 'A sustained tone',
        'musical_details': {'texture': detail('sparse'), 'tempo': detail('120 BPM', False)}, 'reviewed': True}})
    p = store.get(p['id'])
    expected = preview(p)['tracks'][0]['clips'][0]['caption']
    output = tmp_path / 'prepared'
    prepare({'project': p, 'root': str(store.root), 'profile': '8gb', 'prepared_dir': str(output)})
    prepared = [json.loads(f.read_text()) for f in output.glob('*.json') if f.name != 'preparation.json']
    assert prepared[0]['text'] == expected == 'A sustained tone\nTexture: sparse.'
    assert prepared[0]['caption_format'] == FORMAT_VERSION

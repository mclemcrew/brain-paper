import copy

import pytest

from ravi.local.metadata import crop_plan, edit_track, preview, readiness, training_permitted
from ravi.local.store import Store


def track():
    return dict(id='a', name='Piano', family='piano', split='train', caption='Soft piano',
                duration=20, sample_rate=44100, sha256='audio', reviewed=False)


def test_explicit_permission_does_not_claim_review():
    t = track()
    edit_track(t, {'training_authorized': True})
    assert training_permitted(t) and not t['reviewed']
    assert len(crop_plan(t, 15)['clips']) == 2
    p = dict(id='p', updated=0, profile='8gb', tracks=[t, dict(track(), id='b', name='Held', family='held', split='validation')])
    assert readiness(p)['ready']
    assert readiness(p)['counts']['unverified_authorized_tracks'] == 1
    assert 'unverified' in preview(p)['tracks'][0]['issues'][0]


@pytest.mark.parametrize('change', [{'caption': 'Strings'}, {'sections': []}, {'musical_details': {}}])
def test_edit_invalidates_permission(change):
    t = track()
    edit_track(t, {'training_authorized': True})
    edit_track(t, change)
    assert not training_permitted(t)
    with pytest.raises(ValueError, match='Review and approve'):
        crop_plan(t, 15)


def test_permission_is_bound_to_source_and_content_even_for_direct_mutation():
    t = track()
    t['training_authorized'] = True  # Legacy boolean alone is not authorization.
    assert not training_permitted(t)
    edit_track(t, {'training_authorized': True})
    for key, value in [('sha256', 'different'), ('caption', 'Changed'), ('duration', 19)]:
        changed = copy.deepcopy(t)
        changed[key] = value
        assert not training_permitted(changed)


def test_store_authorization_preserves_review_status(tmp_path):
    store = Store(tmp_path)
    p = store.create('Test')
    p['tracks'] = [track()]
    store.save(p)
    p = store.update(p['id'], {'authorize_training': ['a']})
    assert training_permitted(p['tracks'][0])
    assert p['tracks'][0]['reviewed'] is False

import pytest
from ravi.local.store import Store


def test_onboarding_survives_restart_without_changing_metadata(tmp_path):
    store = Store(tmp_path)
    p = store.create('Composer')
    store.update(p['id'], {'onboarding': {'enabled': True, 'step': 2}})
    restarted = Store(tmp_path).get(p['id'])
    assert restarted['onboarding'] == {'enabled': True, 'step': 2}
    assert restarted['tracks'] == p['tracks']
    store.update(p['id'], {'onboarding': {'enabled': False, 'step': 2}})
    assert Store(tmp_path).get(p['id'])['onboarding']['enabled'] is False


@pytest.mark.parametrize('value', [None, {}, {'enabled': 'yes', 'step': 1}, {'enabled': True, 'step': -1}, {'enabled': True, 'step': 5}, {'enabled': True, 'step': True}])
def test_invalid_progress_is_rejected_without_saving(tmp_path, value):
    store = Store(tmp_path)
    p = store.create('Composer')
    with pytest.raises(ValueError):
        store.update(p['id'], {'onboarding': value})
    assert 'onboarding' not in store.get(p['id'])

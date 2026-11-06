import json
import threading
from urllib.request import Request, urlopen

import pytest

from ravi.local.metadata import readiness
from ravi.local.jobs import Jobs, worker_progress
from ravi.local.server import AppServer
from ravi.local.store import Store


def project():
    return {'profile': '8gb', 'tracks': [
        {'id': 'a', 'name': 'First', 'family': 'first', 'split': 'train',
         'caption': 'Quiet piano', 'reviewed': True, 'sample_rate': 44100, 'duration': 30},
        {'id': 'b', 'name': 'Second', 'family': 'second', 'split': 'validation'}]}


def test_worker_progress_handles_partial_and_invalid_announcements():
    tail = ('ordinary output\nRAVI {"message":"Preparing First", "progress":0.25}\n'
            'RAVI {"progress":NaN}\nRAVI {broken}\nRAVI []\n'
            'RAVI {"message":"unfinished", "progress":0.5}')
    assert worker_progress(tail) == {'message': 'Preparing First', 'progress': .25}


def test_ready_counts_exclude_held_out_audio():
    result = readiness(project())
    assert result['ready']
    # v5: the 30 s training piece is one whole example; the held-out recording adds none.
    assert result['counts']['training_clips'] == 1
    assert result['counts']['validation_tracks'] == 1


@pytest.mark.parametrize('change, message', [
    ({'reviewed': False}, 'Review and approve'),
    ({'caption': ''}, 'Describe at least'),
    ({'duration': .5}, 'one second'),
    ({'sections': [{'start': 0, 'end': 2, 'caption': ''}]}, 'Describe every section'),
    ({'sections': [{'start': 0, 'end': 2, 'caption': 'Piano', 'needs_review': True}]}, 'changed sections'),
    ({'sections': [{'start': 0, 'end': 40, 'caption': 'Piano'}]}, 'inside the recording'),
])
def test_content_blockers(change, message):
    p = project()
    p['tracks'][0].update(change)
    result = readiness(p)
    assert not result['ready']
    assert any(message in b['message'] and b['track_id'] == 'a' for b in result['blockers'])


def test_empty_and_leaking_splits():
    assert len(readiness({'profile': '8gb', 'tracks': []})['blockers']) == 2
    p = project()
    p['tracks'][1]['family'] = 'first'
    assert any('both splits' in b['message'] for b in readiness(p)['blockers'])


@pytest.mark.parametrize('kind', ['prepare', 'train', 'calibrate'])
def test_rejected_job_has_no_job_directory(tmp_path, kind):
    store = Store(tmp_path)
    p = store.create('Empty collection')
    jobs = Jobs(store, None)
    with pytest.raises(ValueError, match='training recording'):
        jobs.submit({'kind': kind, 'project_id': p['id']})
    assert list(jobs.root.iterdir()) == []


def test_readiness_endpoint_matches_shared_validation(tmp_path):
    server = AppServer(('127.0.0.1', 0), tmp_path)
    p = server.store.create('Empty collection')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        req = Request(f'http://127.0.0.1:{server.server_port}/api/readiness?project_id={p["id"]}',
                      headers={'X-Ravi-Token': server.token})
        with urlopen(req) as response:
            assert json.load(response) == readiness(p)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

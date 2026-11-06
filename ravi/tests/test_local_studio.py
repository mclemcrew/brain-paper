import json
import subprocess
import sys
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import numpy as np
import pytest
import soundfile as sf

from ravi.local.store import Store, family_name, atomic_json
from ravi.local.server import AppServer
from ravi.local.worker import prepare, infer


def collection(tmp_path):
    store = Store(tmp_path / 'studio')
    p = store.create('Chamber sketches')
    for i, name in enumerate(['Love.wav', 'Love alt2.wav', 'River.wav', 'Movement.wav']):
        path = tmp_path / name
        t = np.arange(44100 * 2) / 44100
        sf.write(path, .1*np.sin(2*np.pi*(220+i*30)*t), 44100, subtype='FLOAT')
        store.import_file(p['id'], path)
    return store, store.get(p['id'])


def test_family_split_and_immutable_import(tmp_path):
    store, p = collection(tmp_path)
    assert family_name('Love.wav') == 'love'
    assert family_name('Movement.wav') == 'movement'
    assert family_name('Love alt2.wav') == 'love'
    p = store.split(p['id'])
    train = {t['family'] for t in p['tracks'] if t['split']=='train'}
    val = {t['family'] for t in p['tracks'] if t['split']=='validation'}
    assert train and val and train.isdisjoint(val)
    assert store.import_file(p['id'], tmp_path/'Love.wav')['duplicate']
    with pytest.raises(ValueError): store.asset_path(p['id'], '../../escape')
    original = store.asset_path(p['id'], p['tracks'][0]['path']).read_bytes()
    store.update(p['id'], {'track':{'id':p['tracks'][0]['id'],'caption':'Sparse strings'}})
    assert store.asset_path(p['id'], p['tracks'][0]['path']).read_bytes() == original


def test_preparation_excludes_validation_and_has_provenance(tmp_path):
    store,p = collection(tmp_path)
    p=store.split(p['id'])
    p=store.update(p['id'],{'description':'Quiet chamber textures'})
    for t in p['tracks']:
        p=store.update(p['id'],{'track':{'id':t['id'],'caption':'Quiet chamber textures','reviewed':True}})
    output=tmp_path/'prepared'
    prepare({'project':p,'root':str(store.root),'profile':'8gb','prepared_dir':str(output)})
    meta=[json.loads(x.read_text()) for x in output.glob('*.json') if x.name!='preparation.json']
    assert {x['source_track'] for x in meta} == {t['id'] for t in p['tracks'] if t['split']=='train'}
    assert all(x['prompt']=='Quiet chamber textures' for x in meta)


def test_edit_preserves_every_outside_sample(tmp_path,monkeypatch):
    store,p=collection(tmp_path)
    track=p['tracks'][0]
    jobdir=tmp_path/'job';jobdir.mkdir()
    def renderer(module,args):
        sf.write(args[args.index('--out')+1],np.ones((88200,2),np.float32)*.2,44100,subtype='FLOAT')
    monkeypatch.setattr('ravi.local.worker.runtime',lambda _:None)
    monkeypatch.setattr('ravi.local.worker.invoke',renderer)
    spec={'root':str(store.root),'project_id':p['id'],'model_id':'sa3-small','profile':'8gb','job_dir':str(jobdir),
          'id':'render1','kind':'edit','source_path':track['path'],'prompt':'Strings','seconds':2,'seed':42,'start':.5,'end':1.5}
    infer(spec)
    effective = json.loads((jobdir/'effective-inference.json').read_text())
    assert effective['inference_steps'] == 8 and effective['seconds'] == 2
    assert effective['cfg'] == 1.0 and effective['dit_dtype'] == 'fp16'
    original,sr=sf.read(store.asset_path(p['id'],track['path']),dtype='float32')
    result=json.loads((jobdir/'result.json').read_text())
    edited,_=sf.read(store.asset_path(p['id'],result['path']),dtype='float32')
    assert np.array_equal(original[:22050],edited[:22050])
    assert np.array_equal(original[66150:],edited[66150:])
    assert not np.array_equal(original[30000:50000],edited[30000:50000])


def test_offline_worker_rejects_connections():
    code='from ravi.local.worker import offline; import socket; offline(); socket.create_connection(("example.com",443))'
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode != 0
    assert 'Network access is disabled' in result.stderr


def test_http_auth_origin_ranges_and_session_token(tmp_path):
    ui=tmp_path/'ui';ui.mkdir();(ui/'index.html').write_text('<meta content="__RAVI_SESSION_TOKEN__">')
    server=AppServer(('127.0.0.1',0),tmp_path/'data',ui=ui)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base) as r: assert server.token in r.read().decode()
        with pytest.raises(HTTPError) as exc: urlopen(base+'/api/state')
        assert exc.value.code==403
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(base+'/api/state',headers={'X-Ravi-Token':server.token,'Origin':'https://evil.example'}))
        assert exc.value.code==403
        with urlopen(Request(base+'/api/state',headers={'X-Ravi-Token':server.token})) as r:
            assert json.load(r)['projects']==[]
        p=server.store.create('test')
        wav=tmp_path/'audio.wav';sf.write(wav,np.zeros(4410),44100)
        p=server.store.import_file(p['id'],wav)['project'];a=p['tracks'][0]
        request=Request(base+f'/api/asset?project_id={p["id"]}&asset_id={a["id"]}',headers={'X-Ravi-Token':server.token,'Range':'bytes=0-15'})
        with urlopen(request) as r:
            assert r.status==206 and len(r.read())==16
    finally:server.shutdown();server.server_close();thread.join()


def test_restart_and_partial_checkpoint_handling(tmp_path):
    from ravi.local.jobs import Jobs
    store=Store(tmp_path/'data');p=store.create('Test')
    job={'id':'ab','kind':'train','status':'running','created':0,'project_id':p['id'],'model_id':'sa3-small'}
    atomic_json(store.root/'jobs/ab/job.json',job)
    cp=store.root/'jobs/ab/checkpoints';cp.mkdir();(cp/'step=10.safetensors').write_bytes(b'broken checkpoint')
    jobs=Jobs(store,None)
    assert jobs.list()[0]['status']=='interrupted'
    assert jobs.checkpoints(p['id'])==[]


def test_cancel_before_worker_starts(tmp_path):
    from ravi.local.jobs import Jobs
    store=Store(tmp_path/'data')
    jobs=Jobs(store,None)
    job={'id':'early','status':'cancelling'}
    jobs.active=job
    jobs._run({'kind':'generate','latents_dir':str(tmp_path/'missing')},job,'')
    assert job['status']=='cancelled'
    assert jobs.active is None and jobs.proc is None


def test_numeric_inputs_reject_nonfinite():
    from ravi.local.jobs import number
    for value in ['nan','inf',float('-inf')]:
        with pytest.raises(ValueError):number(value,0,30,'Duration')


def test_compressed_import_keeps_original_and_uses_exact_decoded_frames(tmp_path):
    import shutil
    from ravi.local.store import digest, read_blocks
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:pytest.skip('ffmpeg required for VBR MP3 regression fixture')
    wav=tmp_path/'source.wav';mp3=tmp_path/'variable.mp3'
    t=np.arange(48000*4)/48000
    signal=(.2*np.sin(2*np.pi*440*t)*(t<2)).astype('float32')
    sf.write(wav,signal,48000)
    subprocess.run([ffmpeg,'-v','error','-i',str(wav),'-q:a','4','-write_xing','0',str(mp3)],check=True)
    before=digest(mp3)
    store=Store(tmp_path/'store');p=store.create('Compressed audio')
    track=store.import_file(p['id'],mp3)['project']['tracks'][0]
    assert track['path'].startswith('decoded/')
    assert digest(store.asset_path(p['id'],track['original_path']))==before==digest(mp3)
    with sf.SoundFile(mp3) as f:expected=np.concatenate(list(read_blocks(f)))
    decoded,sr=sf.read(store.asset_path(p['id'],track['path']),dtype='float32',always_2d=True)
    assert np.array_equal(decoded,expected)
    assert track['duration']==len(expected)/sr


def test_render_export_preserves_shape_without_hard_clipping(tmp_path):
    from ravi.local.worker import write_render
    path=tmp_path/'render.wav'
    raw=np.array([[0,.5,2,-2],[0,.25,1,-1]],dtype=np.float32)
    write_render(path,raw,44100)
    decoded,sr=sf.read(path,dtype='float32',always_2d=True)
    assert sr==44100 and np.allclose(decoded,raw.T*.49)
    assert sf.info(path).subtype=='FLOAT'
    levels=json.loads(path.with_suffix('.levels.json').read_text())
    assert levels['raw_sample_peak']==2 and levels['applied_gain']==.49
    with pytest.raises(ValueError):write_render(tmp_path/'bad.wav',np.array([[np.nan]],dtype=np.float32))


def test_suggestions_preserve_edits_and_review(tmp_path):
    store,p=collection(tmp_path);t=p['tracks'][0]
    p=store.update(p['id'],{'track':{'id':t['id'],'caption':'My own description','reviewed':True}})
    revision=p['tracks'][0]['metadata_revision']
    store.apply_suggestions(p['id'],[{'id':t['id'],'sha256':t['sha256'],'caption':'Machine draft','sections':[]}])
    t=store.get(p['id'])['tracks'][0]
    assert t['caption']=='My own description' and t['reviewed']
    assert t['metadata_revision']==revision
    p=store.update(p['id'],{'track':{'id':t['id'],'caption':'Changed direction'}})
    assert not p['tracks'][0]['reviewed']
    assert p['tracks'][0]['metadata_history'][-1]['caption']=='My own description'


def test_sections_reject_overlap_nan_and_missing_descriptions(tmp_path):
    store,p=collection(tmp_path);t=p['tracks'][0]
    for regions in [[{'start':0,'end':1.5},{'start':1,'end':2}], [{'start':0,'end':float('nan')}]]:
        with pytest.raises(ValueError):store.update(p['id'],{'track':{'id':t['id'],'sections':regions}})
    with pytest.raises(ValueError):
        store.update(p['id'],{'track':{'id':t['id'],'sections':[{'start':0,'end':2}], 'reviewed':True}})
    assert not store.get(p['id'])['tracks'][0].get('sections')


def test_section_crops_use_exact_captions_and_exclude_gaps(tmp_path):
    store,p=collection(tmp_path);t=p['tracks'][0]
    p=store.update(p['id'],{'track':{'id':t['id'],'caption':'Whole piece must not leak',
        'sections':[{'start':.1,'end':1.2,'caption':'Only bowed strings','label':'Opening'}],'reviewed':True}})
    for other in p['tracks'][1:]:
        p=store.update(p['id'],{'track':{'id':other['id'],'split':'exclude'}})
    # Restore the first family after excluding its alternate; alternate stays excluded
    # through direct fixture construction so this test isolates preparation boundaries.
    p['tracks'][0]['split']='train'
    output=tmp_path/'sections'
    prepare({'project':p,'root':str(store.root),'profile':'8gb','prepared_dir':str(output)})
    meta=[json.loads(x.read_text()) for x in output.glob('*.json') if x.name!='preparation.json']
    assert len(meta)==1 and meta[0]['text']=='Only bowed strings'
    assert meta[0]['source_start']==.1 and meta[0]['source_end']==1.2
    assert sf.info(next(output.glob('*.wav'))).frames==48510


def test_bulk_approval_is_atomic_and_edits_invalidate_review(tmp_path):
    store,p=collection(tmp_path)
    ids=[t['id'] for t in p['tracks']]
    with pytest.raises(ValueError):store.update(p['id'],{'review_tracks':ids})
    assert not any(t.get('reviewed') for t in store.get(p['id'])['tracks'])
    p=store.update(p['id'],{'append_captions':{'ids':ids,'text':'Quiet strings'}})
    p=store.update(p['id'],{'review_tracks':ids})
    assert all(t['reviewed'] for t in p['tracks'])
    p=store.update(p['id'],{'append_captions':{'ids':ids[:1],'text':'With piano'}})
    assert not p['tracks'][0]['reviewed'] and p['tracks'][1]['reviewed']


def test_analysis_cache_skips_completed_audio(tmp_path,monkeypatch):
    from ravi.local.analysis import analysis_key
    from ravi.local.worker import analyze
    store,p=collection(tmp_path)
    for t in p['tracks']:
        atomic_json(store.project_dir(p['id'])/'analysis'/f'{analysis_key(t)}.json',
                    {'id':t['id'],'sha256':t['sha256'],'caption':'cached'})
    monkeypatch.setattr('ravi.local.analysis.Tagger', lambda _:pytest.fail('Must not reload models on cache hit'))
    jobdir=tmp_path/'job';jobdir.mkdir()
    analyze({'root':str(store.root),'project':p,'project_id':p['id'],'job_dir':str(jobdir)})
    assert len(json.loads((jobdir/'analysis.json').read_text()))==4


def test_preview_and_preparation_share_frame_exact_crops(tmp_path):
    from ravi.local.metadata import preview
    store,p=collection(tmp_path);t=p['tracks'][0]
    p=store.update(p['id'],{'track':{'id':t['id'],'sections':[
        {'start':.123,'end':1.333,'caption':'Quiet strings','label':'A'},
        {'start':1.5,'end':2.,'caption':'Short ending','label':'B'}], 'reviewed':True}})
    for track in p['tracks'][1:]:track['split']='validation'
    plan=preview(p)
    first=plan['tracks'][0]
    assert first['gap_seconds']==pytest.approx(.29)
    assert first['short_tail_seconds']==.5
    assert len(first['clips'])==1
    output=tmp_path/'preview-prepared'
    prepare({'project':p,'root':str(store.root),'profile':'8gb','prepared_dir':str(output)})
    meta=json.loads(next(x for x in output.glob('*.json') if x.name!='preparation.json').read_text())
    clip=first['clips'][0]
    assert (meta['source_start'],meta['source_end'],meta['text'])==(clip['start'],clip['end'],clip['caption'])
    assert plan['training_clips']==1
    assert all(t['split']=='validation' for t in plan['tracks'][1:])


def test_changed_section_descriptions_must_be_confirmed(tmp_path):
    store,p=collection(tmp_path);t=p['tracks'][0]
    edit={'id':t['id'],'sections':[{'start':0,'end':2,'caption':'Inherited wording','needs_review':True}]}
    p=store.update(p['id'],{'track':edit})
    assert p['tracks'][0]['sections'][0]['needs_review']
    with pytest.raises(ValueError,match='inherited descriptions'):
        store.update(p['id'],{'review_tracks':[t['id']]})
    edit['sections'][0]['needs_review']=False
    edit['reviewed']=True
    p=store.update(p['id'],{'track':edit})
    assert p['tracks'][0]['reviewed']


def test_medium_default_preserves_existing_small_collection(tmp_path):
    store = Store(tmp_path)
    project = store.create('Medium pilot')
    assert (project['model_id'], project['profile']) == ('sa3-medium', '16gb')
    store.update(project['id'], {'model_id': 'sa3-small', 'profile': '8gb'})
    restored = Store(tmp_path).get(project['id'])
    assert (restored['model_id'], restored['profile']) == ('sa3-small', '8gb')

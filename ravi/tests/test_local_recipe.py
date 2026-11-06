import json

import pytest

from ravi.local import worker
from ravi.local.catalog import crop_latents
from ravi.local.metadata import crop_plan
from ravi.local.recipe import EXCLUDED_TARGETS, TRIGGER, training_recipe
from ravi.local.jobs import Jobs
from ravi.local.store import Store


def patched_runtime(tmp_path, patched=True):
    scripts = tmp_path / 'runtime' / 'scripts'
    scripts.mkdir(parents=True)
    (scripts / 'lora_train_mlx.py').write_text('ap.add_argument("--gen-mask")' if patched else '# upstream')
    return str(tmp_path / 'runtime')


def train_args(tmp_path, monkeypatch, profile='8gb', steps=1000):
    calls = []
    monkeypatch.setattr(worker, 'runtime', lambda _: None)
    monkeypatch.setattr(worker, 'invoke', lambda name, args: calls.append((name, args)))
    recipe = training_recipe(profile)
    worker.train(dict(profile=profile, recipe=recipe, job_dir=str(tmp_path), runtime=patched_runtime(tmp_path),
                      model_id='sa3-small' if profile == '8gb' else 'sa3-medium', latents_dir='latents', id='test',
                      checkpoint_root='checkpoints', steps=steps, seed=42))
    name, args = calls[0]
    assert name == 'lora_train_mlx'
    return recipe, args


def value(args, flag):
    return args[args.index(flag) + 1]


def test_v5_training_command_matches_cloud_recipe(tmp_path, monkeypatch):
    recipe, args = train_args(tmp_path, monkeypatch)
    assert value(args, '--gen-mask') == 'zeros'
    assert value(args, '--exclude').split(',') == list(EXCLUDED_TARGETS) == ['seconds_total']
    assert value(args, '--timestep-sampler') == 'trunc_logit_normal'
    assert '--use-effective-length' in args and '--no-use-effective-length' not in args
    assert '--random-crop' in args
    assert (value(args, '--beta2'), value(args, '--weight-decay')) == (.95, .01)
    assert '--no-conditioner-lora' in args
    assert value(args, '--batch-size') == recipe['batch_size']
    assert value(args, '--gradient-clip-val') == recipe['gradient_clip_val'] == 0  # cloud: no clipping
    prompt = json.loads((tmp_path / 'prompt.json').read_text())
    assert prompt['trigger'] == TRIGGER and prompt['trigger_pct'] == 100


@pytest.mark.parametrize('profile,frames', [('8gb', 161), ('16gb', 322), ('32gb', 506)])
def test_crop_is_whole_latent_frames(tmp_path, monkeypatch, profile, frames):
    assert crop_latents(profile) == frames
    assert frames * 4096 <= training_recipe(profile)['crop_seconds'] * 44100
    _, args = train_args(tmp_path, monkeypatch, profile)
    assert value(args, '--latent-crop-length') == frames
    assert value(args, '--rank') == training_recipe(profile)['rank']


@pytest.mark.parametrize('steps,every', [(1000, 250), (6000, 500)])
def test_checkpoint_spacing(tmp_path, monkeypatch, steps, every):
    _, args = train_args(tmp_path, monkeypatch, steps=steps)
    assert value(args, '--checkpoint-every') == every


def test_training_refuses_unpatched_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'runtime', lambda _: pytest.fail('Loaded runtime before checking the patch'))
    with pytest.raises(RuntimeError, match='generation-mask patch'):
        worker.train(dict(profile='8gb', recipe=training_recipe('8gb'), job_dir=str(tmp_path),
                          runtime=patched_runtime(tmp_path, patched=False), model_id='sa3-small'))


@pytest.mark.parametrize('recipe', [None, 'local-mlx-fullmask-v3', {'name': 'old'},
                                    {'name': 'local-mlx-fullmask-v4-protected-targets'}])
def test_worker_rejects_incompatible_continuation_before_loading_models(recipe, monkeypatch):
    def forbidden(_):
        pytest.fail('Loaded runtime before validating continuation recipe')
    monkeypatch.setattr(worker, 'runtime', forbidden)
    with pytest.raises(ValueError, match='recipe differs'):
        worker.train(dict(profile='8gb', recipe=recipe, adapter_path='old.safetensors'))


def test_examples_are_whole_regions_split_evenly_without_tails():
    t = dict(name='Spot', caption='Bright brass', duration=32.3, sample_rate=48000, reviewed=True)
    plan = crop_plan(t, 120, require_review=False)
    assert len(plan['clips']) == 1 and plan['clips'][0]['end_frame'] == round(32.3 * 48000)
    long = dict(t, duration=250.0)
    clips = crop_plan(long, 120, require_review=False)['clips']
    lengths = [c['end_frame'] - c['start_frame'] for c in clips]
    assert len(clips) == 3 and max(lengths) - min(lengths) <= 1 and clips[-1]['end_frame'] == 250 * 48000


def configured_jobs(tmp_path, monkeypatch):
    store = Store(tmp_path)
    p = store.create('Test')
    p = store.update(p['id'], {'model_id': 'sa3-small', 'profile': '8gb'})
    monkeypatch.setattr('ravi.local.jobs.catalog', lambda *a: [{'id':'sa3-small', 'ready':True}])
    monkeypatch.setattr('ravi.local.jobs.hardware', lambda: {'memory_gb':48, 'available_gb':30})
    monkeypatch.setattr('ravi.local.jobs.threading.Thread.start', lambda _: None)
    monkeypatch.setattr('ravi.local.metadata.readiness', lambda _: {'ready':True})
    return Jobs(store, None), p


def test_generation_records_actual_sampling_steps(tmp_path, monkeypatch):
    jobs, p = configured_jobs(tmp_path, monkeypatch)
    job = jobs.submit({'project_id':p['id'], 'kind':'generate'})
    spec = json.loads((jobs.root/job['id']/'request.json').read_text())
    assert job['steps'] == job['inference_steps'] == spec['steps'] == spec['inference_steps'] == 8
    assert 'trigger' not in spec


@pytest.mark.parametrize('trained_recipe,expected', [(training_recipe('8gb'), TRIGGER),
                                                     ({'name': 'local-mlx-fullmask-v4-protected-targets'}, None)])
def test_generation_inherits_trigger_from_training_recipe(tmp_path, monkeypatch, trained_recipe, expected):
    jobs, p = configured_jobs(tmp_path, monkeypatch)
    (jobs.root / 'trainjob').mkdir()
    (jobs.root / 'trainjob' / 'job.json').write_text(json.dumps({'recipe': trained_recipe}))
    monkeypatch.setattr(jobs, 'checkpoints', lambda _: [dict(id='cp', model_id='sa3-small', job_id='trainjob',
                        cache_key='k', profile='8gb', step=1000, path='cp.safetensors')])
    job = jobs.submit({'project_id':p['id'], 'kind':'generate', 'adapter_id':'cp', 'prompt':'Warm piano'})
    spec = json.loads((jobs.root/job['id']/'request.json').read_text())
    assert spec['trigger'] == expected


def test_old_recipe_checkpoint_cannot_continue(tmp_path, monkeypatch):
    jobs, p = configured_jobs(tmp_path, monkeypatch)
    monkeypatch.setattr(jobs, 'checkpoints', lambda _: [dict(id='old', model_id='sa3-small',
                        cache_key='old-v3-hash', profile='8gb', step=250, path='old.safetensors')])
    with pytest.raises(ValueError, match='collection or recipe changed'):
        jobs.submit({'project_id':p['id'], 'kind':'train', 'adapter_id':'old'})


@pytest.mark.parametrize('strength,expected', [(1, 'rvkr, Warm piano'), (0, 'Warm piano')])
def test_inference_trigger_only_when_adapter_is_active(tmp_path, monkeypatch, strength, expected):
    calls = []
    monkeypatch.setattr(worker, 'runtime', lambda _: None)
    monkeypatch.setattr(worker, 'invoke', lambda name, args: calls.append(args))
    monkeypatch.setattr(worker, 'publish_audio', lambda *a: None)
    worker.infer(dict(job_dir=str(tmp_path), model_id='sa3-small', profile='8gb', kind='generate',
                      prompt='Warm piano', seconds=15, seed=1, adapter_path='a.safetensors',
                      strength=strength, trigger=TRIGGER))
    assert value(calls[0], '--prompt') == expected
    record = json.loads((tmp_path / 'effective-inference.json').read_text())
    assert record['prompt_sent'] == expected


def test_submit_refuses_unpatched_trainer_before_creating_work(tmp_path, monkeypatch):
    jobs, p = configured_jobs(tmp_path, monkeypatch)
    jobs.runtime = patched_runtime(tmp_path, patched=False)
    with pytest.raises(ValueError, match='generation-mask fix'):
        jobs.submit({'project_id': p['id'], 'kind': 'train'})

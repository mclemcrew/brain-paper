"""Composer-owned metadata and exact section-to-caption training alignment."""
import copy
import hashlib
import json
import math
import time
import uuid


DETAIL_FIELDS = {'instruments': 'Instruments', 'vocals': 'Vocals', 'mood': 'Mood',
                 'texture': 'Texture', 'techniques': 'Playing techniques', 'tempo': 'Tempo'}
FORMAT_VERSION = 'musical-details-v1'


def metadata_fingerprint(track):
    """Bind draft permission to the audio and the exact conditioning metadata."""
    value = {key: track.get(key) for key in
             ('sha256', 'duration', 'sample_rate', 'caption', 'sections', 'musical_details')}
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def training_permitted(track):
    authorization = track.get('training_authorization')
    return track.get('reviewed') is True or (
        isinstance(authorization, dict)
        and authorization.get('metadata_sha256') == metadata_fingerprint(track)
        and authorization.get('scope') == 'unverified-metadata')


def musical_details(value):
    if not isinstance(value, dict):
        raise ValueError('Musical details must be an object.')
    result = {}
    for key, item in value.items():
        if key not in DETAIL_FIELDS or not isinstance(item, dict):
            raise ValueError('Unknown musical detail.')
        text = str(item.get('value', '')).strip()[:500]
        if not text and not item.get("dismissed"): continue
        result[key] = {'value': text, 'reviewed': item.get('reviewed') is True,
                       'source': str(item.get('source', 'Composer'))[:200],
                       'dismissed': item.get('dismissed') is True}
        if isinstance(item.get('suggestion'), dict):
            result[key]['suggestion'] = copy.deepcopy(item['suggestion'])
    return result


def conditioning_text(caption, details):
    parts = [str(caption or '').strip()]
    for key, label in DETAIL_FIELDS.items():
        item = musical_details(details or {}).get(key)
        if item and item['reviewed'] and item['value'] and not item['dismissed']:
            parts.append(f"{label}: {item['value']}.")
    return '\n'.join(p for p in parts if p)


def sections(value, duration):
    if not isinstance(value, list) or len(value) > 256:
        raise ValueError('Use at most 256 sections per piece.')
    result = []
    for s in value:
        start, end = float(s['start']), float(s['end'])
        if not all(map(math.isfinite, (start, end))) or not 0 <= start < end <= duration:
            raise ValueError('Section times must fall inside the recording, with start before end.')
        result.append({'id': str(s.get('id') or uuid.uuid4().hex)[:64],
                       'start': start, 'end': end, 'label': str(s.get('label', 'Section'))[:120],
                       'caption': str(s.get('caption', '')).strip()[:2000],
                       'needs_review': s.get('needs_review') is True,
                       'musical_details': musical_details(s.get('musical_details', {}))})
    result.sort(key=lambda s: s['start'])
    if any(a['end'] > b['start'] for a, b in zip(result, result[1:])):
        raise ValueError('Sections must not overlap. Gaps are allowed and are excluded from training.')
    if len({s['id'] for s in result}) != len(result):
        raise ValueError('Section identifiers must be unique.')
    return result


def edit_track(track, edit):
    """A suggestion never changes approved text. Content edits invalidate approval."""
    before = copy.deepcopy(track)
    for key in ('caption', 'family', 'bpm', 'split'):
        if key in edit: track[key] = str(edit[key]).strip()[:2000]
    if 'musical_details' in edit:
        track['musical_details'] = musical_details(edit['musical_details'])
    if 'sections' in edit:
        track['sections'] = sections(edit['sections'], track['duration'])
    if 'accept_suggestion' in edit:
        suggestion = track.get('suggestion', {})
        if edit['accept_suggestion'] == 'caption':
            if not suggestion.get('caption'): raise ValueError('No caption suggestion is available.')
            track['caption'] = suggestion['caption']
        elif edit['accept_suggestion'] == 'sections':
            track['sections'] = sections(suggestion.get('sections', []), track['duration'])
        else: raise ValueError('Unknown suggestion.')
    content_changed = any(before.get(k) != track.get(k) for k in ('caption', 'sections', 'musical_details'))
    if content_changed:
        track['reviewed'] = False
        track.pop('training_authorization', None)
        track['metadata_source'] = 'Composer draft'
    if 'reviewed' in edit:
        if edit['reviewed'] and any(s.get('needs_review') for s in track.get('sections', [])):
            raise ValueError('Check the inherited descriptions for changed sections before approving.')
        if edit['reviewed'] and not training_regions(track, '', require_review=False):
            raise ValueError('Add a description for the piece or its sections before approving.')
        track['reviewed'] = edit['reviewed'] is True
        if track['reviewed']: track['metadata_source'] = 'Reviewed by composer'
    if 'training_authorized' in edit:
        if type(edit['training_authorized']) is not bool:
            raise ValueError('Training permission must be true or false.')
        if edit['training_authorized']:
            if not training_regions(track, '', require_review=False):
                raise ValueError('Add a description before authorizing training.')
            track['training_authorization'] = {
                'scope': 'unverified-metadata', 'metadata_sha256': metadata_fingerprint(track),
                'authorized_at': time.time()}
        else:
            track.pop('training_authorization', None)
    if not track['family'] or track['split'] not in ('train', 'validation', 'exclude'):
        raise ValueError('Choose a family and a valid split.')
    if track != before:
        history = track.setdefault('metadata_history', [])
        history.append({k: copy.deepcopy(before.get(k)) for k in
                        ('caption', 'sections', 'musical_details', 'reviewed', 'training_authorization', 'metadata_source', 'metadata_revision')})
        track['metadata_revision'] = int(before.get('metadata_revision', 0)) + 1
        track['metadata_updated'] = time.time()


def training_regions(track, description='', require_review=True):
    if require_review and not training_permitted(track):
        raise ValueError(f"Review and approve the metadata for {track['name']} before training.")
    if require_review and any(s.get('needs_review') for s in track.get('sections', [])):
        raise ValueError('Check descriptions for changed sections before training.')
    if track.get('sections'):
        result = sections(track['sections'], track['duration'])
        for region in result:
            region['caption'] = conditioning_text(region['caption'], region.get('musical_details'))
        if any(not s['caption'] for s in result):
            raise ValueError(f"Describe every section of {track['name']}; section captions replace the piece description.")
        return result
    caption = conditioning_text(track.get('caption', '').strip() or description.strip(), track.get('musical_details'))
    if not caption: return []
    return [{'id': 'whole', 'start': 0., 'end': track['duration'], 'label': 'Whole piece', 'caption': caption}]


def crop_plan(track, max_seconds, require_review=True):
    """Single frame-based source of truth for preview and preparation.

    Each described region (whole piece, or each timed section) is one training example, so
    examples keep their real beginnings and endings; the trainer samples a random window from
    each. Regions longer than ``max_seconds`` are split into equal parts (never a short tail).
    Parts shorter than one second are discarded.
    """
    sr = int(track['sample_rate'])
    clips = []
    regions = training_regions(track, require_review=require_review)
    covered = 0
    discarded = 0
    limit = int(max_seconds * sr)
    for region in regions:
        first, last = round(region['start'] * sr), round(region['end'] * sr)
        covered += last-first
        parts = max(1, -(-(last-first) // limit))
        bounds = [first + round(i * (last-first) / parts) for i in range(parts + 1)]
        for position, end in zip(bounds, bounds[1:]):
            if end-position < sr:
                discarded += end-position
                continue
            clips.append({'start_frame': position, 'end_frame': end,
                          'start': position/sr, 'end': end/sr,
                          'section_id': region['id'], 'label': region['label'], 'caption': region['caption'],
                          'caption_format': FORMAT_VERSION})
    return {'clips': clips, 'gap_seconds': max(0, round(track['duration']*sr)-covered)/sr,
            'short_tail_seconds': discarded/sr}


def preview(project):
    from .catalog import PROFILES
    crop = PROFILES[project['profile']]['crop_seconds']
    example_max = PROFILES[project['profile']]['example_max_seconds']
    tracks = []
    for t in project['tracks']:
        item = {'id':t['id'], 'name':t['name'], 'split':t['split'], 'reviewed':bool(t.get('reviewed')),
                'training_authorized': training_permitted(t),
                'clips':[], 'gap_seconds':0, 'short_tail_seconds':0, 'issues':[]}
        if t['split'] == 'exclude':
            item['issues'].append('Entire recording excluded.')
        else:
            try:
                item.update(crop_plan(t, example_max, require_review=False))
                if not item['clips']: item['issues'].append('No described audio clips of at least one second.')
            except (ValueError, TypeError) as exc:
                item['issues'].append(str(exc))
            if any(s.get('needs_review') for s in t.get('sections', [])):
                item['issues'].append('Changed sections need their descriptions checked.')
            if not t.get('reviewed'):
                item['issues'].append('Training authorized with unverified metadata.' if training_permitted(t)
                                      else 'Metadata needs approval.')
        tracks.append(item)
    return {'project_id':project['id'], 'updated':project['updated'], 'crop_seconds':crop,
            'example_max_seconds':example_max, 'tracks':tracks,
            'training_clips':sum(len(t['clips']) for t in tracks if t['split']=='train'),
            'note':'Source excerpts and exact captions. Preparation resamples and normalizes audio; silent clips may be omitted.'}


def readiness(project):
    """Collection-only prerequisites shared by the UI and job submission.

    This checks described source crops, not audibility or model availability.
    Validation audio is reserved for manual listening and is not prepared.
    """
    from .catalog import PROFILES
    tracks = project.get('tracks', [])
    train = [t for t in tracks if t.get('split') == 'train']
    validation = [t for t in tracks if t.get('split') == 'validation']
    blockers = []
    def block(message, track=None):
        item = {'message': message}
        if track is not None: item['track_id'] = track['id']
        blockers.append(item)
    if not train: block('Keep at least one training recording.')
    if not validation: block('Reserve at least one validation recording for comparison.')
    shared = {t.get('family') for t in train} & {t.get('family') for t in validation}
    if shared: block('An alternate family appears in both splits.')
    example_max = PROFILES[project['profile']]['example_max_seconds']
    clip_count = 0
    for track in tracks:
        if track.get('split') not in ('train', 'validation', 'exclude'):
            block('Choose a valid recording split.', track)
        if track.get('split') != 'exclude' and not str(track.get('family', '')).strip():
            block('Choose a piece family.', track)
    for track in train:
        # Check approval and content separately so all actionable issues appear.
        if not training_permitted(track):
            block(f"Review and approve the metadata for {track['name']} before training.", track)
        if any(s.get('needs_review') for s in track.get('sections', [])):
            block('Check descriptions for changed sections before training.', track)
        try:
            clips = crop_plan(track, example_max, require_review=False)['clips']
            clip_count += len(clips)
            if not clips:
                block(f"Describe at least one passage of one second or longer in {track['name']}.", track)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            block(str(exc), track)
    return {'project_id': project.get('id'), 'ready': not blockers, 'blockers': blockers,
            'counts': {'tracks': len(tracks), 'train_tracks': len(train),
                       'validation_tracks': len(validation),
                       'excluded_tracks': sum(t.get('split') == 'exclude' for t in tracks),
                       'unverified_authorized_tracks': sum(training_permitted(t) and not t.get('reviewed') for t in train),
                       'training_clips': clip_count}}

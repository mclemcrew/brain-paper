from ravi.local.server import export_name


def test_export_names_distinguish_original_and_personal_and_are_header_safe():
    asset = {'name': 'Quiet piano, soft strings', 'path': 'renders/abc.wav', 'id': 'abcdef1234', 'kind': 'generate'}
    assert export_name(asset) == 'quiet-piano-soft-strings-generate-original-abcdef12.wav'
    assert 'personal' in export_name({**asset, 'adapter_id': 'cp'})
    hostile = export_name({**asset, 'name': '../../café"\r\nX-Header: injected', 'kind': 'edit\r\n', 'id': 'x"\r\n'})
    assert all(c not in hostile for c in '\r\n"/\\')
    assert hostile.endswith('.wav')


def test_midi_export_has_midi_extension_and_empty_title_fallback():
    assert export_name({'name': '音', 'path': 'renders/a.mid', 'id': '123', 'kind': 'transcribe'}).endswith('-transcribe-original-123.mid')

"""Additional immutable recovery identity; never rewrite the original STUDY."""
from pathlib import Path
import json
import os
from .compat import content_sha

REPO=Path('/opt/data/private/longtail_vlm/rarevlm_pilot/packages/RareVLM_minimal_20261001')
PROTOCOL=REPO/'docs/VLM_OBJECT_BEHAVIOR_REPLAY_RECOVERY_V1_PROTOCOL.md'
REVIEW_SHA='e0cf20c0c0bde47243aff542dc2fc763da726e67f9de8f8a38fa67cae6aa4935'


def sha_file(file):
    import hashlib
    digest=hashlib.sha256()
    with Path(file).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def read(file):return json.loads(Path(file).read_bytes())


def verify_files(mapping):
    for name,digest in mapping.items():
        if not Path(name).is_file() or sha_file(name)!=digest:raise ValueError(f'recovery-bound file changed: {name}')


def source_hashes():
    files=[*Path(__file__).parent.glob('*.py'),PROTOCOL,
        REPO/'tools/recover_object_behavior_replay_pilot.py',REPO/'tests/test_object_behavior_recovery.py']
    return {str(file.resolve()):sha_file(file) for file in sorted(files)}


def active_original_processes(config_file):
    """Do not duplicate a surviving controller, worker or trainer of this config."""
    found=[];config=str(Path(config_file).resolve()).encode()
    for folder in Path('/proc').iterdir():
        if not folder.name.isdecimal() or int(folder.name)==os.getpid():continue
        try:
            command=(folder/'cmdline').read_bytes();raw=(folder/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if fields[0]=='Z':continue
        if config in command and any(name in command for name in (
            b'vlm_object_behavior_replay.study',b'vlm_object_behavior_replay.train',
            b'run_object_behavior_replay_study.py')):
            found.append({'pid':int(folder.name),'start_ticks':fields[19],'state':fields[0]})
    return found


def assert_original_inactive(config_file):
    found=active_original_processes(config_file)
    if found:raise ValueError(f'original processes remain active; never duplicate or terminate them: {found}')


def verify_lock(file,config_file):
    from vlm_object_behavior_replay.study import verify_study
    file,config_file=Path(file).resolve(),Path(config_file).resolve();lock=read(file)
    if (lock.get('schema_version')!='rarevlm.object_behavior_replay.recovery_input.v1'
        or lock.get('config_file')!=str(config_file) or lock.get('config_file_sha256')!=sha_file(config_file)
        or lock.get('recovery_source_sha256')!=source_hashes()
        or lock.get('scope')!='completed_seed42_Finals_to_full_internal_pilot_only'
        or lock.get('new_training_or_teacher_generation_allowed') is not False):
        raise ValueError('effective recovery source/config/scope lock differs')
    original=verify_study(Path(lock['original_study_input_lock_file']),config_file)
    if (sha_file(lock['original_study_input_lock_file'])!=lock['original_study_input_lock_sha256']
        or original['source_sha256']!=lock['original_scientific_source_sha256']
        or original['files_sha256']!=lock['original_scientific_files_sha256']):
        raise ValueError('original scientific lock changed; recovery does not revise it')
    verify_files(lock['recovery_source_sha256']);verify_files(lock['files_sha256'])
    return lock

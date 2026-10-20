"""Inspect or restore artifacts listed in results/large-files.csv.

From the SMQ repository root: python script/restore_result_artifacts.py --restore
Without --restore, only inventory the manifest. Existing files are never overwritten.
"""
import argparse
import csv
import os
import stat
from pathlib import Path, PurePosixPath

def is_link(path):
    """Recognize symlinks and Windows junctions on Python 3.10 and later."""
    try:
        info=path.lstat()
    except FileNotFoundError:
        return False
    return path.is_symlink() or bool(getattr(info,'st_file_attributes',0) & getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0x400))

def inside(relative, repo, expected_prefix):
    parts=PurePosixPath(relative).parts
    if not parts or parts[0]!=expected_prefix or '..' in parts or '\\' in relative:
        raise ValueError(f'Invalid manifest path: {relative}')
    target=repo.joinpath(*parts)
    target.parent.resolve().relative_to((repo/expected_prefix).resolve())
    return target

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore',action='store_true',help='Move the archived artifacts back to their original paths.')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    results=repo/'results'
    with (results/'large-files.csv').open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    pending=[]
    restored=missing=0
    for row in rows:
        if row['large_path']!='large/'+row['original_path']:
            raise ValueError('Archive path does not mirror the original path.')
        original=inside(row['original_path'],repo,'results')
        archive=inside(row['large_path'],repo,'large')
        link_record=row.get('entry_type','file') in ('junction','symlink')
        available=is_link(archive) if link_record else archive.is_file()
        if available:
            if original.exists() or is_link(original):
                raise FileExistsError(f'Both copies exist; refusing to overwrite {original}')
            if link_record and os.readlink(archive)!=row['link_target']:
                raise ValueError(f'Archive link target differs from manifest: {archive}')
            if not link_record and archive.stat().st_size!=int(row['size_bytes']):
                raise ValueError(f'Archive size differs from manifest: {archive}')
            pending.append((original,archive))
        elif original.is_file() or is_link(original):
            restored+=1
        else:
            missing+=1
    print(f'{len(pending)} archived files available, {restored} already restored, {missing} absent from this checkout.',flush=True)
    if not args.restore:
        return
    if missing:
        raise FileNotFoundError('Supply the local large/ folder before restoring the full experiment artifacts.')
    for i,(original,archive) in enumerate(pending,1):
        if original.exists():
            raise FileExistsError(original)
        original.parent.mkdir(parents=True,exist_ok=True)
        os.rename(archive,original)
        if i%5000==0 or i==len(pending):
            print(f'Restored {i}/{len(pending)} files.',flush=True)

if __name__=='__main__':
    main()

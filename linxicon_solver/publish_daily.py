"""Prepare a daily Pages artifact, reusing the last complete published replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time

import requests

from .daily import fetch_game
from .data import DATA_DIR
from .export_daily import SCHEMA_VERSION, build_daily, load_bundle, rejected_words, revision, write_bundle
from .server import ServerClient
from .vocab import load_rejected

REJECTED_SCHEMA=1


def load_rejected_list(output):
    """The website's accumulated list of words the game rejected (empty if absent or invalid)."""
    path=Path(output)/'rejected.json'
    try:
        data=json.loads(path.read_text())
        if data.get('schema_version')!=REJECTED_SCHEMA:
            return set()
        return {w for w in data.get('words',[]) if isinstance(w,str) and w.isalpha() and w.islower()}
    except (OSError,ValueError):
        return set()


def update_rejected(output,*,seed,learned):
    """Merge the seed list, the previously published list and today's rejections; returns the union."""
    output=Path(output)
    words=load_rejected_list(output)|set(seed)|set(learned)
    data=dict(schema_version=REJECTED_SCHEMA,words=sorted(words))
    output.mkdir(parents=True,exist_ok=True)
    temporary=output/'rejected.tmp'
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(output/'rejected.json')
    return words


def update(output, game, solver_revision, generate, *, force=False):
    output=Path(output)
    old=None
    if (output/'latest.json').exists():
        load_bundle(output)
        old=json.loads((output/'latest.json').read_text())
    same=old and old['game']['id']==game.id and old['solver_revision']==solver_revision and old['schema_version']==SCHEMA_VERSION
    attempts=old.get('attempts',1) if same else 0
    if same and not force:
        if old['verification_status']!='unavailable':
            return dict(changed=False,outcome='unchanged',game_id=game.id)
        if attempts>=3:
            return dict(changed=False,outcome='retry_limit',game_id=game.id)
    if old and old['game']['id']>game.id and not force:
        return dict(changed=False,outcome='stale',error='Puzzle source returned an older ID; retaining the newer published result.')
    try:
        result=generate(game)
        result['solver_revision']=solver_revision
        write_bundle(result,output,attempts=attempts+1)
    except Exception as exc:
        if not old:
            raise
        return dict(changed=False,outcome='stale',error=str(exc),game_id=old['game']['id'])
    return dict(changed=True,outcome='generated',game_id=game.id,verification_status=result['verification_status'])


def restore_published(base_url, output):
    """Validate remote files in isolation before replacing the bootstrap snapshot."""
    base_url=base_url.rstrip('/')+'/'
    try:
        response=requests.get(base_url+'latest.json',timeout=30)
        response.raise_for_status()
        manifest=response.json()
        name=manifest['file']
        if Path(name).name!=name or not name.startswith('daily-') or not name.endswith('.json'):
            raise ValueError('Invalid published bundle filename.')
        response=requests.get(base_url+name,timeout=30)
        response.raise_for_status()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)
            (path/'latest.json').write_text(json.dumps(manifest))
            (path/name).write_bytes(response.content)
            load_bundle(path)
            output.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path/name,output/name)
            shutil.copyfile(path/'latest.json',output/'latest.json')
        try:
            response=requests.get(base_url+'rejected.json',timeout=30)
            response.raise_for_status()
            with tempfile.TemporaryDirectory() as temporary:
                path=Path(temporary)
                (path/'rejected.json').write_bytes(response.content)
                words=load_rejected_list(path)
            if words:
                update_rejected(output,seed=words,learned=set())
        except (requests.RequestException,OSError) as exc:
            print(f'Published rejected list unavailable: {exc}')
        return True
    except (requests.RequestException,ValueError,KeyError,OSError) as exc:
        print(f'Published data unavailable; using checked-in snapshot: {exc}')
        return False


def generate(game,blocked=frozenset()):
    from scripts.setup_data import main as setup
    from .data import prepare
    start=time.perf_counter()
    if setup([])!=0:
        raise RuntimeError('Dataset setup failed.')
    sim,graph=prepare((game.tl,game.br),progress=print)
    result=build_daily(game,sim,graph,ServerClient(),revision=revision(),blocked=blocked)
    result['timings']['total']=time.perf_counter()-start
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous-url')
    parser.add_argument('--status',type=Path,required=True)
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args(argv)
    if args.previous_url:
        restore_published(args.previous_url,args.output)
    seed=load_rejected(DATA_DIR/'rejected_words.txt')
    try:
        game=fetch_game()
        blocked=update_rejected(args.output,seed=seed,learned=set())
        status=update(args.output,game,revision(),lambda game:generate(game,blocked),force=args.force)
        if status['changed']:
            learned=rejected_words(load_bundle(args.output))
            status['rejected_words_learned']=sorted(learned-blocked)
            update_rejected(args.output,seed=seed,learned=learned)
    except Exception as exc:
        # A complete bootstrap or previously published artifact still permits
        # unrelated website updates when the game's endpoint is unavailable.
        load_bundle(args.output)
        status=dict(changed=False,outcome='stale',error=str(exc))
    args.status.parent.mkdir(parents=True,exist_ok=True)
    args.status.write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())

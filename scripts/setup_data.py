"""Download the fixed datasets and prepare the same caches used by the CLI."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow `python scripts/setup_data.py` from a source checkout.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nltk
import requests

from linxicon_solver.cli import add_vocab_arguments, resolve_min_zipf
from linxicon_solver.data import DATA_DIR, NUMBERBATCH_NAME, prepare
from linxicon_solver.similarity import NUMBERBATCH_URL
from linxicon_solver.vocab import ENABLE_URL


def download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.part')
    print(f'Downloading {destination.name}…', file=sys.stderr)
    try:
        with requests.get(url, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            with temporary.open('wb') as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        if temporary.stat().st_size == 0:
            raise ValueError(f'Empty download for {destination.name}.')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_vocab_arguments(parser)
    args = parser.parse_args(argv)
    min_zipf = resolve_min_zipf(parser, args)
    try:
        download(ENABLE_URL, DATA_DIR/'enable1.txt')
        download(NUMBERBATCH_URL, DATA_DIR/NUMBERBATCH_NAME)
        if not args.no_wordnet:
            if not nltk.download('wordnet', quiet=True, raise_on_error=True):
                raise ValueError('WordNet download failed; check connectivity and NLTK_DATA permissions.')
        prepare(data_dir=DATA_DIR, min_zipf=min_zipf, use_wordnet=not args.no_wordnet,
                progress=lambda message: print(message, file=sys.stderr))
    except (OSError, ValueError, LookupError, requests.RequestException) as exc:
        print(f'Setup failed: {exc}', file=sys.stderr)
        return 1
    print('Data and graph cache are ready.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

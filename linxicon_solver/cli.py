"""Command-line search, board simulation, and optional server verification."""
from __future__ import annotations

import argparse
import math
import sys

from .board import Board
from .daily import fetch_game
from .data import DATA_DIR, prepare
from .rules import MAX_WORDS, THRESHOLD
from .server import MissingVectorError, ServerClient, ServerError, WordRejectedError
from .vocab import DEFAULT_MIN_ZIPF, is_valid_word, load_rejected


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def zipf_value(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError('must be a finite, nonnegative number')
    return number


def add_vocab_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--vocab', choices=('common', 'full'), default='common',
                        help='common uses the frequency floor; full includes all valid ENABLE words')
    parser.add_argument('--min-zipf', type=zipf_value, help=f'common vocabulary frequency floor (default: {DEFAULT_MIN_ZIPF})')
    parser.add_argument('--no-wordnet', action='store_true', help='use Numberbatch cosine alone')


def resolve_min_zipf(parser, args) -> float:
    if args.vocab == 'full' and args.min_zipf is not None:
        parser.error('--vocab full cannot be combined with --min-zipf')
    return 0.0 if args.vocab == 'full' else (DEFAULT_MIN_ZIPF if args.min_zipf is None else args.min_zipf)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Find shortest Linxicon chains in a local vocabulary and scoring model.')
    parser.add_argument('words', nargs='*', help='two starter words')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--today', action='store_true', help="solve today's puzzle")
    modes.add_argument('--game', type=positive_int, help='solve a numbered puzzle')
    parser.add_argument('--alternates', type=positive_int, default=5, help='maximum total candidates (default: 5)')
    add_vocab_arguments(parser)
    parser.add_argument('--verify', action='store_true', help='validate displayed chains and replay server scores')
    parser.add_argument('--no-simulate', action='store_true', help='skip local and server board replay')
    args = parser.parse_args(argv)
    min_zipf = resolve_min_zipf(parser, args)
    if args.today or args.game is not None:
        if args.words:
            parser.error('provide either two words, --today, or --game ID')
    elif len(args.words) != 2:
        parser.error('provide two words, --today, or --game ID')
    if args.words:
        starters = tuple(w.lower() for w in args.words)
        if any(not is_valid_word(w) for w in starters):
            parser.error('starters must contain 3–15 ASCII letters')
        if starters[0] == starters[1]:
            parser.error('starters must be distinct')
    try:
        if not args.words:
            game = fetch_game(args.game)
            starters = (game.tl, game.br)
            print(f'Game #{game.id} ({game.date}): {game.tl} → {game.br}; published similarity: {game.similarity:.6f}')
        sim, graph = prepare(starters, min_zipf=min_zipf, use_wordnet=not args.no_wordnet,
                             progress=lambda message: print(message, file=sys.stderr))
        for word in starters:
            if word not in sim.vectors:
                raise ValueError(f'Starter "{word}" has no Numberbatch vector.')
        blocked = load_rejected(DATA_DIR / 'rejected_words.txt')
        chains = graph.shortest_chains(*starters, k=args.alternates, blocked=blocked)
        if not chains:
            raise ValueError('No path in this vocabulary and scoring model. Try --vocab full or a lower --min-zipf.')
        if len(chains[0].words) > MAX_WORDS:
            raise ValueError(f'Shortest chain exceeds the {MAX_WORDS}-word board cap.')
        print('Ranked by fewest added words, then highest total local score.')
        client = ServerClient() if args.verify else None
        successful = False
        for rank, chain in enumerate(chains, 1):
            print(f'\n{rank}. ' + ' → '.join(chain.words))
            print(f'   added: {chain.added}; total: {chain.total:.6f}; average: {chain.average:.6f}; weakest: {chain.weakest:.6f}')
            for a, b, score in zip(chain.words, chain.words[1:], chain.scores):
                print(f'   {a} → {b}: local {score:.6f}')
            local_won = True
            if args.no_simulate:
                print('   local board: SKIPPED')
            else:
                result = Board(starters, sim.score).simulate(chain.words[1:-1])
                local_won = result.won
                print(f'   local board: {"WIN" if result.won else "FAILED"}; added: {result.words_added}; path: ' + ' → '.join(result.path))
            if client is None:
                successful |= local_won
                continue
            try:
                verified = client.verify(chain.words)
            except (WordRejectedError, MissingVectorError) as exc:
                print(f'   server: REJECTED — {exc}')
                continue
            for a, b, local, remote in zip(chain.words, chain.words[1:], chain.scores, verified.scores):
                print(f'   {a} → {b}: local {local:.6f}; server {remote:.6f}')
            links_valid = all(score >= THRESHOLD for score in verified.scores)
            print(f'   server words: ACCEPTED; chain links: {"PASS" if links_valid else "FAILED"}')
            if args.no_simulate:
                print('   server board: SKIPPED')
                successful |= links_valid
            else:
                replay = Board(starters, verified.score).simulate(chain.words[1:-1])
                print(f'   server board: {"WIN" if replay.won else "FAILED"}; server added: {replay.words_added}; path: ' + ' → '.join(replay.path))
                successful |= replay.won
        return 0 if successful else 1
    except (OSError, ValueError, LookupError, ServerError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

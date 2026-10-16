"""Concurrency choices bounded by the selected package's declared capacity."""


def add_arguments(parser, defaults):
    parser.add_argument("--max-sequences", type=int, default=32,
                        help="Selected package's declared maximum sequences (default 32)")
    parser.add_argument("--concurrencies", type=int, nargs="+", default=list(defaults),
                        help="Unique increasing concurrency values, bounded by --max-sequences")


def validate(parser, args):
    if args.max_sequences < 1:
        parser.error("max-sequences must be positive")
    if args.concurrencies != sorted(set(args.concurrencies)) or any(
            value < 1 or value > args.max_sequences for value in args.concurrencies):
        parser.error("concurrencies must be unique, increasing values from 1 through max-sequences")

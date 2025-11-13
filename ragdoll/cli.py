import argparse
import sys

from ragdoll.ingestion import list_loaders


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ragdoll")
    parser.add_argument("--list-loaders", action="store_true", help="List registered loader short names")
    args = parser.parse_args(argv)

    if args.list_loaders:
        loaders = list_loaders()
        if not loaders:
            print("No loaders registered")
            return 0
        for name in loaders:
            print(name)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

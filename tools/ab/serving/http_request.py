#!/usr/bin/env python3
"""One HTTP read, run under the supervisor's absolute process deadline."""
import sys
import urllib.error
import urllib.request


def main():
    url, timeout = sys.argv[1:]
    try:
        with urllib.request.urlopen(url, timeout=float(timeout)) as response:
            if response.status != 200:
                return 3
            body = response.read(4 * 1024 * 1024 + 1)
            if len(body) > 4 * 1024 * 1024:
                print('HTTP response exceeds limit', file=sys.stderr)
                return 2
        sys.stdout.buffer.write(body)
        return 0
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        print(str(error), file=sys.stderr)
        return 3


if __name__ == '__main__':
    sys.exit(main())

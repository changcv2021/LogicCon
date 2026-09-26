"""Bounded-memory input and reproducible provenance, using only the standard library."""
import hashlib
import json
import os
import socket
from pathlib import Path


def compute_guard():
    if socket.gethostname().split('.')[0].startswith('login'):
        raise SystemExit('Bulk work is disabled on login nodes. Use an approved Slurm allocation or transfer host.')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def jsonl(path):
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def dictionary_items(stream, chunk_size=65536):
    """Stream a top-level JSON dictionary from a text file or ZipExtFile wrapper.

    Never load all of GQA. Memory is bounded by a single source record (max 16 MiB).
    """
    decoder = json.JSONDecoder()
    buf, eof = '', False

    def more():
        nonlocal buf, eof
        block = stream.read(chunk_size)
        eof = not block
        buf += block
        if len(buf) > 16 * 1024 * 1024:
            raise ValueError('Oversized/malformed source record')

    def whitespace():
        nonlocal buf
        while True:
            buf = buf.lstrip()
            if buf or eof:
                return
            more()

    def value():
        nonlocal buf
        whitespace()
        while True:
            try:
                out, end = decoder.raw_decode(buf)
                buf = buf[end:]
                return out
            except json.JSONDecodeError:
                if eof:
                    raise
                more()

    whitespace()
    if not buf.startswith('{'):
        raise ValueError('Expected a top-level object')
    buf = buf[1:]
    whitespace()
    if buf.startswith('}'):
        return
    while True:
        key = value()
        if not isinstance(key, str):
            raise ValueError('Non-string source key')
        whitespace()
        if not buf.startswith(':'):
            raise ValueError('Expected colon')
        buf = buf[1:]
        yield key, value()
        whitespace()
        if buf.startswith('}'):
            buf = buf[1:]
            whitespace()
            if buf:
                raise ValueError('Trailing JSON data')
            return
        if not buf.startswith(','):
            raise ValueError('Expected comma or end of object')
        buf = buf[1:]

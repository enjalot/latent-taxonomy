#!/usr/bin/env python3
"""Tiny CORS-enabled static server for locally-hosted model sample shards.

The site reads `samples_base_url_local` from a model's metadata.json when
running on localhost, so big sample sets that live outside web/public (e.g.
/data/latent-taxonomy/<MODEL>/samples) can be served during development
without copying ~1GB into the repo.

Usage:
  python3 pipeline/serve_samples.py --root /data/latent-taxonomy --port 8787
  # -> http://localhost:8787/MINILM_STAGEJ_49K/samples/chunk_0.parquet
"""
import argparse
import functools
from http.server import HTTPServer, SimpleHTTPRequestHandler


class CORSHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/latent-taxonomy")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    handler = functools.partial(CORSHandler, directory=args.root)
    print(f"serving {args.root} at http://localhost:{args.port}/ (CORS *)")
    HTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()

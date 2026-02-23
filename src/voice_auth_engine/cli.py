"""voice-auth-engine CLI エントリポイント。"""

import argparse

from voice_auth_engine.model_downloader import ModelDownloader


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="voice-auth-engine")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("download-models", help="全モデルをダウンロードする")

    args = parser.parse_args(argv)

    if args.command == "download-models":
        ModelDownloader().download_all()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

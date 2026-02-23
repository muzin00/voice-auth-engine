"""Download model files for sherpa-onnx."""

from voice_auth_engine.model_downloader import ModelDownloader


def main() -> None:
    ModelDownloader().download_all()


if __name__ == "__main__":
    main()

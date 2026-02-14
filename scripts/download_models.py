"""Download model files for sherpa-onnx."""

from pathlib import Path

from voice_auth_engine.model_downloader import ModelDownloader

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    downloader = ModelDownloader(models_dir=MODELS_DIR)
    downloader.download_all()


if __name__ == "__main__":
    main()

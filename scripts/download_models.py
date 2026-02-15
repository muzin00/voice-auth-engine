"""Download model files for sherpa-onnx."""

from voice_auth_engine.model_config import MODELS_DIR
from voice_auth_engine.model_downloader import ModelDownloader


def main() -> None:
    downloader = ModelDownloader(models_dir=MODELS_DIR)
    downloader.download_all()


if __name__ == "__main__":
    main()

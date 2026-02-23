"""Download model files for sherpa-onnx."""

import logging
import shutil
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

from voice_auth_engine.model_config import DEFAULT_MODELS, ModelConfig

logger = logging.getLogger(__name__)


class ModelDownloadError(Exception):
    """モデルのダウンロードに失敗した場合の例外。"""


class ModelDownloader:
    """モデルファイルをダウンロードする。"""

    def __init__(
        self, models_dir: Path | None = None, models: list[ModelConfig] | None = None
    ) -> None:
        self.models_dir = models_dir or ModelConfig.get_models_dir()
        self.models = models or DEFAULT_MODELS

    def download_all(self) -> None:
        """全モデルをダウンロードする。"""
        print(f"Models directory: {self.models_dir}")
        print()

        for model in self.models:
            self.download(model)

        print("All models downloaded.")

    def download(self, model: ModelConfig) -> None:
        """単一モデルをダウンロードする。すでにダウンロード済みならスキップ。"""
        print(f"[{model.name}]")
        if self.is_downloaded(model):
            print("  Already downloaded, skipping.")
            print()
            return

        dest = self.models_dir / model.dest
        print(f"  Downloading from {model.url}")
        if model.archive:
            assert model.inner_dir is not None
            self._download_and_extract_tar(model.url, dest, model.inner_dir)
        else:
            self._download_file(model.url, dest)
        print("  Done.")
        print()

    def is_downloaded(self, model: ModelConfig) -> bool:
        """モデルがダウンロード済みかどうかを判定する。"""
        dest = self.models_dir / model.dest
        if model.archive:
            return dest.exists() and any(dest.iterdir())
        return dest.exists()

    def _download_with_progress(self, url: str, dest: Path) -> None:
        response = urllib.request.urlopen(url)  # noqa: S310
        total_size = int(response.headers.get("Content-Length", 0))
        block_size = 8192

        with tqdm(total=total_size, unit="B", unit_scale=True) as progress:
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    progress.update(len(chunk))

    def _download_file(self, url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._download_with_progress(url, dest)

    def _download_and_extract_tar(self, url: str, dest: Path, inner_dir: str) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        tmp_file = dest / "_download.tar.bz2"
        try:
            self._download_with_progress(url, tmp_file)
            print("  Extracting...")
            with tarfile.open(tmp_file, "r:bz2") as tar:
                tar.extractall(path=dest, filter="data")  # noqa: S202
            # Move files from inner directory to dest
            inner_path = dest / inner_dir
            if inner_path.exists():
                for item in inner_path.iterdir():
                    shutil.move(str(item), str(dest / item.name))
                inner_path.rmdir()
        finally:
            tmp_file.unlink(missing_ok=True)

    def ensure_download(self, model: ModelConfig) -> Path:
        """モデルが存在することを保証し、パスを返す。

        ダウンロード済みでなければ自動的にダウンロードする。

        Args:
            model: モデル設定。

        Returns:
            モデルファイル/ディレクトリの絶対パス。

        Raises:
            ModelDownloadError: ダウンロードに失敗した場合。
        """
        if self.is_downloaded(model):
            return self.models_dir / model.dest

        logger.info("モデル '%s' をダウンロードしています...", model.name)
        try:
            self.download(model)
        except Exception as exc:
            raise ModelDownloadError(
                f"モデル '{model.name}' のダウンロードに失敗しました: {exc}"
            ) from exc

        return self.models_dir / model.dest

"""Kokoro TTS で数字の音声ファイルを生成する。"""

import argparse
from pathlib import Path

import soundfile as sf
from kokoro_onnx import Kokoro

from voice_auth_engine.model_config import kokoro_tts_config, kokoro_voices_config

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
DEFAULT_OUTPUT = OUTPUTS_DIR / "output.wav"


def digits_to_text(number: str) -> str:
    """数値文字列をスペース区切りの数字に変換する。"""
    digits = [d for d in number if d.isdigit()]
    if not digits:
        return ""
    return " ".join(digits)


def main() -> None:
    parser = argparse.ArgumentParser(description="数字を日本語で読み上げた音声を生成する")
    parser.add_argument("number", help="読み上げる数値 (例: 1234)")
    parser.add_argument("--voice", default="jf_alpha", help="音声名 (デフォルト: jf_alpha)")
    parser.add_argument("--speed", type=float, default=1.0, help="話速 (デフォルト: 1.0)")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="出力ファイルパス (デフォルト: output.wav)",
    )
    args = parser.parse_args()

    text = digits_to_text(args.number)
    if not text:
        parser.error("数値に有効な数字 (0-9) が含まれていません")

    print(f"テキスト: {text}")
    kokoro = Kokoro(str(kokoro_tts_config.path), str(kokoro_voices_config.path))
    audio, sample_rate = kokoro.create(
        text=text,
        voice=args.voice,
        speed=args.speed,
        lang="ja",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), audio, sample_rate)
    print(f"生成完了: {args.output}")


if __name__ == "__main__":
    main()

"""Tests for CLI entry point."""

from unittest.mock import patch

from voice_auth_engine.cli import main


class TestCli:
    def test_download_models_calls_downloader(self):
        with patch("voice_auth_engine.cli.ModelDownloader") as mock_cls:
            main(["download-models"])
            mock_cls.return_value.download_all.assert_called_once()

    def test_no_command_prints_help(self, capsys):
        main([])
        captured = capsys.readouterr()
        assert "download-models" in captured.out

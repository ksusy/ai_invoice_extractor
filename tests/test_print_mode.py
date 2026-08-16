"""Tests for PRINT_MODE toggle."""

from __future__ import annotations

from unittest.mock import patch

from src.config.settings import PrintMode
from src.domain.constants import PRINT_MODE


class TestPrintMode:
    def test_default_is_color(self):
        assert PRINT_MODE() == PrintMode.COLOR

    def test_grayscale_override(self):
        with patch("src.domain.constants.get_settings") as mock:
            mock.return_value.print_mode = PrintMode.GRAYSCALE
            assert PRINT_MODE() == PrintMode.GRAYSCALE

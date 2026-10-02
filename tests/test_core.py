"""Unit tests for core helpers."""

# pylint: disable=missing-function-docstring
from unittest.mock import MagicMock

import core


def make_file(name, text):
    file = MagicMock()
    file.name = name
    file.getvalue.return_value = text.encode("utf-8")
    return file


def test_process_prompt_expands_file():
    assert "professional email" in core.process_prompt("@[templates/email.txt]")


def test_process_prompt_refuses_dotfiles_and_escapes(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("SECRET=1")
    (tmp_path / "sub").mkdir()
    (tmp_path / "outside.txt").write_text("outside")
    monkeypatch.chdir(tmp_path / "sub")

    assert core.process_prompt("@[../.env]") == "[Access denied: ../.env]"
    assert core.process_prompt("@[../outside.txt]") == "[Access denied: ../outside.txt]"
    assert core.process_prompt("@[/etc/passwd]") == "[Access denied: /etc/passwd]"


def test_process_prompt_missing_and_truncated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "big.txt").write_text("x" * (core.MAX_INCLUDE_BYTES + 10))

    assert core.process_prompt("@[nope.txt]") == "[File not found: nope.txt]"
    assert core.process_prompt("@[big.txt]").endswith("[truncated]")


def test_build_user_message_keeps_file_content():
    message = core.build_user_message("Summarise this", [make_file("notes.txt", "launch is on Friday")])

    assert "launch is on Friday" in message["content"]
    assert "launch is on Friday" not in message["display"]
    assert "notes.txt" in message["display"]


def test_build_payload_uses_stored_content():
    history = [
        {"role": "user", "content": "expanded", "display": "short"},
        {"role": "assistant", "content": "reply"},
    ]
    payload = core.build_payload(history, "be brief")

    assert payload == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "expanded"},
        {"role": "assistant", "content": "reply"},
    ]


def test_model_help_ollama_and_display_name():
    provider = MagicMock()
    provider.get_name.return_value = "OpenRouter"

    provider.get_model_info.return_value = {"size": 2 * 1024**3, "details": {"parameter_size": "7B"}}
    assert "2.00 GB" in core.model_help(provider, "qwen")

    provider.get_model_info.return_value = {"details": {"display_name": "Auto (Free)"}}
    help_text = core.model_help(provider, "openrouter/auto")
    assert "Auto (Free)" in help_text
    assert "GB" not in help_text


def test_text_chunks_skips_empty():
    stream = [{"message": {"content": "a"}}, {"message": {"content": ""}}, {"message": {"content": "b"}}]
    assert list(core.text_chunks(stream)) == ["a", "b"]


def test_markdown_export_uses_display_text():
    messages = [{"role": "user", "content": "huge file dump", "display": "short"}]
    exported = core.format_chat_as_markdown(messages)
    assert "short" in exported
    assert "huge file dump" not in exported

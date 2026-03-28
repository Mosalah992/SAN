"""Tests for reply_handler post field coercion and relevance helper."""

from reply_handler import _is_relevant, _post_field_as_text


class TestPostFieldAsText:
    def test_none_empty(self):
        assert _post_field_as_text(None) == ""

    def test_str_unchanged(self):
        assert _post_field_as_text("hello") == "hello"

    def test_dict_json(self):
        s = _post_field_as_text({"k": "security", "n": 1})
        assert "security" in s
        assert "k" in s

    def test_list_json(self):
        s = _post_field_as_text(["a", "b"])
        assert "a" in s

    def test_int_str(self):
        assert _post_field_as_text(42) == "42"


class TestIsRelevantNoCrash:
    def test_dict_shaped_title_and_content(self):
        post = {
            "title": {"headline": "nothing"},
            "content": {"body": "philosophy ethics consciousness debate"},
            "submolt_name": "m/philosophy",
        }
        _is_relevant(post)

    def test_dict_submolt(self):
        post = {
            "title": "hello world",
            "content": "more text",
            "submolt": {"name": "m/security"},
        }
        _is_relevant(post)

import os
import unittest
from unittest.mock import patch


class ProviderConfigTests(unittest.TestCase):
    def test_lmstudio_uses_local_openai_compatible_defaults(self):
        from providers import get_provider_config

        with patch.dict(os.environ, {"MOA_PROVIDER": "lmstudio"}, clear=True):
            config = get_provider_config()

        self.assertEqual(config.provider, "lmstudio")
        self.assertEqual(config.base_url, "http://127.0.0.1:1234/v1")
        self.assertEqual(config.api_key, "lm-studio")

    def test_omp_requires_a_base_url_and_prefers_omp_api_key(self):
        from providers import get_provider_config

        env = {
            "MOA_PROVIDER": "omp",
            "MOA_BASE_URL": "http://127.0.0.1:4141/v1",
            "OMP_API_KEY": "omp-secret",
        }
        with patch.dict(os.environ, env, clear=True):
            config = get_provider_config()

        self.assertEqual(config.provider, "omp")
        self.assertEqual(config.base_url, "http://127.0.0.1:4141/v1")
        self.assertEqual(config.api_key, "omp-secret")

    def test_provider_config_repr_does_not_reveal_api_key(self):
        from providers import get_provider_config

        env = {
            "MOA_PROVIDER": "omp",
            "MOA_BASE_URL": "http://127.0.0.1:4141/v1",
            "OMP_API_KEY": "super-secret",
        }
        with patch.dict(os.environ, env, clear=True):
            config = get_provider_config()

        self.assertNotIn("super-secret", repr(config))

    def test_atomic_agents_provider_is_an_explicit_placeholder(self):
        from providers import generate_chat_completion

        with patch.dict(os.environ, {"MOA_PROVIDER": "atomic"}, clear=True):
            with self.assertRaisesRegex(NotImplementedError, "Atomic Agents"):
                generate_chat_completion(
                    model="placeholder",
                    messages=[{"role": "user", "content": "hello"}],
                )


class _StatusError(Exception):
    def __init__(self, status):
        super().__init__(f"http {status}")
        self.status_code = status


class RetryClassificationTests(unittest.TestCase):
    def test_is_retryable_distinguishes_transient_from_permanent(self):
        from providers import _is_retryable

        self.assertTrue(_is_retryable(_StatusError(503)))
        self.assertTrue(_is_retryable(_StatusError(429)))
        self.assertTrue(_is_retryable(ConnectionError("reset")))
        self.assertFalse(_is_retryable(_StatusError(404)))
        self.assertFalse(_is_retryable(_StatusError(401)))
        self.assertFalse(_is_retryable(ValueError("bad config")))

    def test_non_retryable_error_raises_immediately_without_sleeping(self):
        import providers

        with patch("providers.generate_chat_completion", side_effect=ValueError("bad model")), patch(
            "providers.time.sleep"
        ) as sleep:
            with self.assertRaises(ValueError):
                providers.generate_text_with_retries(
                    model="nope", messages=[{"role": "user", "content": "x"}]
                )
            sleep.assert_not_called()

    def test_retryable_error_exhausts_then_reraises(self):
        import providers

        with patch(
            "providers.generate_chat_completion", side_effect=_StatusError(503)
        ), patch("providers.time.sleep") as sleep:
            with self.assertRaises(_StatusError):
                providers.generate_text_with_retries(
                    model="m", messages=[{"role": "user", "content": "x"}], max_retries=3
                )
            # Slept between attempts but not after the final one: 3 attempts -> 2 sleeps.
            self.assertEqual(sleep.call_count, 2)


class EmptyChoicesGuardTests(unittest.TestCase):
    def test_get_completion_text_handles_empty_choices(self):
        from providers import get_completion_text

        class _Resp:
            choices = []

        self.assertEqual(get_completion_text(_Resp()), "")

    def test_stream_text_chunks_skips_empty_choices(self):
        from providers import stream_text_chunks

        class _Chunk:
            def __init__(self, choices):
                self.choices = choices

        class _Delta:
            content = "hi"

        class _Choice:
            delta = _Delta()

        chunks = [_Chunk([]), _Chunk([_Choice()])]
        self.assertEqual("".join(stream_text_chunks(chunks)), "hi")


class ImageGenerationTests(unittest.TestCase):
    def test_generate_image_returns_b64(self):
        import providers

        class _Item:
            b64_json = "QUJD"
            url = None

        class _Resp:
            data = [_Item()]

        class _Client:
            class images:
                @staticmethod
                def generate(**kwargs):
                    return _Resp()

        with patch("providers._sync_client_from_config", return_value=_Client()):
            out = providers.generate_image(
                "a cat", model="sd", provider="openai-compatible", base_url="http://x/v1"
            )
        self.assertEqual(out[0]["b64_json"], "QUJD")

    def test_generate_image_falls_back_when_response_format_rejected(self):
        import providers

        calls = []

        class _Item:
            b64_json = None
            url = "http://img/1.png"

        class _Resp:
            data = [_Item()]

        class _Client:
            class images:
                @staticmethod
                def generate(**kwargs):
                    calls.append(kwargs)
                    if "response_format" in kwargs:
                        raise TypeError("unsupported response_format")
                    return _Resp()

        with patch("providers._sync_client_from_config", return_value=_Client()):
            out = providers.generate_image("a cat", model="sd", provider="openai-compatible", base_url="http://x/v1")
        self.assertEqual(out[0]["url"], "http://img/1.png")
        self.assertEqual(len(calls), 2)  # first with response_format, retry without

    def test_generate_image_atomic_placeholder(self):
        import providers

        with patch.dict(os.environ, {"MOA_PROVIDER": "atomic"}, clear=True):
            with self.assertRaisesRegex(NotImplementedError, "Atomic"):
                providers.generate_image("x")


class ClientCacheTests(unittest.TestCase):
    def test_sync_clients_are_reused_for_same_config(self):
        import providers

        providers._sync_client_cache.clear()
        # Mock the constructor so the test doesn't build a real httpx/SSL client.
        constructed = []

        def fake_openai(**kwargs):
            obj = object()
            constructed.append(obj)
            return obj

        with patch.dict(os.environ, {"MOA_PROVIDER": "lmstudio"}, clear=True), patch(
            "providers.openai.OpenAI", side_effect=fake_openai
        ):
            first = providers.create_client()
            second = providers.create_client()

        self.assertIs(first, second)
        self.assertEqual(len(constructed), 1)  # built once, reused on the second call

    def test_sync_client_construction_sets_timeout_and_no_internal_retries(self):
        import providers

        providers._sync_client_cache.clear()
        captured = {}

        def fake_openai(**kwargs):
            captured.update(kwargs)
            return object()

        with patch.dict(os.environ, {"MOA_PROVIDER": "lmstudio"}, clear=True), patch(
            "providers.openai.OpenAI", side_effect=fake_openai
        ):
            providers.create_client()

        self.assertEqual(captured.get("max_retries"), 0)
        self.assertIn("timeout", captured)


if __name__ == "__main__":
    unittest.main()

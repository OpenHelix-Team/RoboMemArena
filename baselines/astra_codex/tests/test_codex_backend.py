"""Local protocol tests: a fake child process, no network, model, or robot."""

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codex_backend import CodexBackend, CodexBackendError, SharedCodexRuntime


def fake_server(scenario):
    """Exercise wire ordering and process failure, not a mocked robot result."""
    assert "OPENAI_API_KEY" not in os.environ
    assert "CODEX_API_KEY" not in os.environ
    threads = turns = 0

    def send(value):
        print(json.dumps(value), flush=True)

    for line in sys.stdin:
        message = json.loads(line)
        method, params = message.get("method"), message.get("params", {})
        if "id" not in message:
            continue
        result = {}
        if method == "config/read":
            result = {"config": {"model_provider": "openai", "mcp_servers": {
                "test_mcp": {"env": {"SECRET": "private-config-value"}}}}}
            if scenario == "custom-provider":
                result["config"]["model_providers"] = {"openai": {"base_url": "https://example.invalid"}}
        elif method == "account/read":
            result = {"account": {"type": "apiKey" if scenario == "api-key" else "chatgpt",
                                  "planType": "pro", "email": "private@example.test"},
                      "requiresOpenaiAuth": True}
        elif method == "thread/start":
            assert params["model"] == "gpt-6-astra"
            assert params["ephemeral"] is True
            assert params['config']['features.multi_agent'] is (scenario == 'ultra')
            assert params["environments"] == params["dynamicTools"] == []
            assert params["config"]["mcp_servers.test_mcp.enabled"] is False
            threads += 1
            result = {"thread": {"id": f"thread-{threads}"}, "model": params["model"]}
        elif method == "turn/start":
            assert params["effort"] == ('ultra' if scenario == 'ultra' else 'medium')
            turns += 1
            turn_id, thread_id = f"turn-{turns}", params["threadId"]
            if scenario == "disconnect":
                return
            if scenario == "timeout":
                continue
            result = {"turn": {"id": turn_id, "status": "inProgress"}}
            if scenario == "tool-request":
                send({"id": message["id"], "result": result})
                send({"id": 999, "method": "item/tool/call", "params": {"name": "shell"}})
                continue
            if scenario == "failed":
                send({"id": message["id"], "result": result})
                send({"method": "turn/completed", "params": {"threadId": thread_id,
                      "turn": {"id": turn_id, "status": "failed", "error": {"message": "limit reached"}}}})
                continue
            # A fast child can finish items before acknowledging turn/start.
            answer = json.dumps({"thread": thread_id, "turn": turns})
            send({"method": "item/completed", "params": {"threadId": thread_id, "turnId": turn_id,
                  "item": {"type": "agentMessage", "phase": "final_answer", "text": answer}}})
            send({"method": "thread/tokenUsage/updated", "params": {"threadId": thread_id,
                  "turnId": turn_id, "tokenUsage": {"last": {"totalTokens": 3}}}})
            send({"id": message["id"], "result": result})
            send({"method": "turn/completed", "params": {"threadId": "another-thread",
                  "turn": {"id": turn_id, "status": "failed"}}})
            send({"method": "turn/completed", "params": {"threadId": thread_id,
                  "turn": {"id": turn_id, "status": "completed"}}})
            continue
        send({"id": message["id"], "result": result})


class CodexBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.real_popen = subprocess.Popen

    def backend(self, scenario="ok", timeout=3):
        def popen(command, **kwargs):
            self.assertIn('forced_login_method="chatgpt"', command)
            return self.real_popen([sys.executable, __file__, "--fake-server", scenario], **kwargs)

        self.addCleanup(patch.stopall)
        patch("codex_backend.subprocess.Popen", side_effect=popen).start()
        patch("codex_backend.subprocess.run",
              return_value=subprocess.CompletedProcess([], 0, "codex-cli test\n", "")).start()
        backend = CodexBackend(self.root, codex_bin=sys.executable, timeout_s=timeout,
                               reasoning_effort='ultra' if scenario == 'ultra' else 'medium')
        self.addCleanup(backend.close)
        return backend

    def test_ultra_keeps_ephemeral_history_and_enables_only_native_delegation(self):
        backend = self.backend('ultra')
        backend.start('Return JSON')
        self.assertEqual(json.loads(backend.ask('Observe'))['turn'], 1)
        self.assertTrue(backend._overrides['features.multi_agent'])
        self.assertTrue(all(value is False for key, value in backend._overrides.items()
                            if key.startswith('features.') and key != 'features.multi_agent'))

    def test_shared_close_after_router_thread_could_not_start(self):
        runtime = SharedCodexRuntime(self.root, codex_bin=sys.executable)
        runtime._router = threading.Thread(target=lambda: None)
        runtime.close()
        self.assertIsNone(runtime._owner.process)

    def test_shared_transient_thread_limit_does_not_relaunch_transport(self):
        runtime = SharedCodexRuntime(self.root, codex_bin=sys.executable)
        runtime._owner._launch = Mock()
        runtime._owner._check_auth = Mock()
        router = Mock()
        router.start.side_effect = [RuntimeError("can't start new thread"), None]
        with patch('codex_backend.threading.Thread', return_value=router), \
             patch('codex_backend.time.sleep') as pause:
            self.assertIs(runtime.start(), runtime)
        runtime._owner._launch.assert_called_once()
        runtime._owner._check_auth.assert_called_once()
        pause.assert_called_once_with(1)
        runtime.close()

    def test_history_reset_image_order_and_private_trace(self):
        backend = self.backend()
        image = self.root / "frame.png"
        image.write_bytes(b"fake image; the protocol test never performs vision")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "must-not-reach-child", "CODEX_API_KEY": "must-not-reach-child"}):
            backend.start("Return one JSON action from the supplied observations")
        first = json.loads(backend.ask(content=[{"type": "text", "text": "Frame 1"},
                                               {"type": "localImage", "path": str(image)}]))
        second = json.loads(backend.ask("Next observation"))
        self.assertEqual(first["thread"], second["thread"])
        self.assertEqual((first["turn"], second["turn"]), (1, 2))
        self.assertEqual(backend.last_turn["usage"]["last"]["totalTokens"], 3)
        backend.reset("New episode")
        self.assertNotEqual(first["thread"], json.loads(backend.ask("Observe"))["thread"])
        trace_text = (self.root / "codex_protocol.jsonl").read_text()
        for secret in ("private@example.test", "private-config-value", "must-not-reach-child"):
            self.assertNotIn(secret, trace_text)
        turns = [row["message"]["params"] for row in map(json.loads, trace_text.splitlines())
                 if row["direction"] == "send" and row["message"].get("method") == "turn/start"]
        self.assertEqual([item["type"] for item in turns[0]["input"]], ["text", "localImage"])
        self.assertEqual(turns[0]["input"][1]["path"], str(image.resolve()))
        self.assertEqual({turn["model"] for turn in turns}, {"gpt-6-astra"})

    def test_api_key_login_rejected_before_turn(self):
        backend = self.backend("api-key")
        with self.assertRaisesRegex(CodexBackendError, "ChatGPT login required"):
            backend.start()
        self.assertIsNone(backend.process)
        self.assertNotIn('"method": "turn/start"', (self.root / "codex_protocol.jsonl").read_text())

    def test_custom_provider_rejected(self):
        backend = self.backend("custom-provider")
        with self.assertRaisesRegex(CodexBackendError, "Custom OpenAI provider"):
            backend.start()
        self.assertIsNone(backend.process)

    def test_disconnect_is_not_replayed(self):
        backend = self.backend("disconnect").start()
        with self.assertRaisesRegex(CodexBackendError, "disconnected"):
            backend.ask("Observation")
        self.assertIsNone(backend.process)
        self.assertEqual(backend.last_turn["status"], "error")
        with self.assertRaisesRegex(CodexBackendError, "No active episode"):
            backend.ask("No silent restart")
        trace = (self.root / "codex_protocol.jsonl").read_text()
        self.assertEqual(trace.count('"method": "turn/start"'), 1)

    def test_timeout_closes_child(self):
        backend = self.backend("timeout", timeout=1).start()
        with self.assertRaises(TimeoutError):
            backend.ask("Observation")
        self.assertIsNone(backend.process)

    def test_model_failure_is_not_an_answer(self):
        backend = self.backend("failed").start()
        with self.assertRaisesRegex(CodexBackendError, "limit reached"):
            backend.ask("Observation")
        self.assertIsNone(backend.process)

    def test_unexpected_tool_request_is_rejected(self):
        backend = self.backend("tool-request").start()
        with self.assertRaisesRegex(CodexBackendError, "Unexpected server request"):
            backend.ask("Observation")
        self.assertIsNone(backend.process)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fake-server":
        fake_server(sys.argv[2])
    else:
        unittest.main()

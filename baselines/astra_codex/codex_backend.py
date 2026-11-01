"""Subscription-only, persistent Codex app-server conversations for evaluation.

One backend serves one episode at a time. ``start(instructions)`` or ``reset``
creates a fresh thread; successive ``ask`` calls retain its native history.
Answers are returned unchanged for the evaluator to parse. Transport failures
never replay a turn or silently replace its history: record the failed episode,
then explicitly reset before continuing.

Protocol reference: https://developers.openai.com/codex/app-server
No credentials are read, copied, or supplied by this module.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
from typing import Any, Iterable


class CodexBackendError(RuntimeError):
    """Authentication, protocol, or model-turn failure; no API fallback exists."""


class CodexBackend:
    def __init__(
        self,
        trace_dir: str | Path,
        *,
        codex_bin: str | None = None,
        model: str = "gpt-6-astra",
        reasoning_effort: str = "medium",
        timeout_s: float = 180,
        runtime: SharedCodexRuntime | None = None,
    ) -> None:
        self.codex_bin = shutil.which(codex_bin or os.environ.get("CODEX_BIN", "codex"))
        if not self.codex_bin:
            raise FileNotFoundError("Codex binary not found; pass codex_bin or set CODEX_BIN")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.model, self.reasoning_effort, self.timeout_s = model, reasoning_effort, timeout_s
        self.trace_dir = Path(trace_dir).resolve()
        self.process: subprocess.Popen[str] | None = None
        self.thread_id: str | None = None
        self.last_turn: dict[str, Any] = {}
        self.auth: dict[str, Any] = {}
        self.version: str | None = None
        self._trace = self._stderr = None
        self._log_lock = threading.Lock()
        self._runtime = runtime
        self._next_id = 0
        self._requests: dict[int, str] = {}
        self._pending: deque[dict[str, Any]] = deque()
        self._events: queue.Queue = queue.Queue()
        self._overrides = {
            "model_provider": "openai",
            "forced_login_method": "chatgpt",
            "model_reasoning_effort": reasoning_effort,
            "web_search": "disabled",
            "project_doc_max_bytes": 0,
            **{f"features.{name}": False for name in (
                "shell_tool", "unified_exec", "view_image", "browser_use",
                "browser_use_external", "computer_use", "apps", "plugins",
                "multi_agent", "code_mode", "code_mode_host", "skill_search",
                "workspace_dependencies", "hooks", "memories", "goals",
            )},
        }
        self._overrides['features.multi_agent'] = reasoning_effort == 'ultra'

    def start(self, instructions: str = "") -> "CodexBackend":
        """Start app-server, verify ChatGPT login, and create an episode thread."""
        if self.process is not None:
            raise CodexBackendError("Already started; use reset for a new episode")
        return self.reset(instructions)

    def reset(self, instructions: str = "", trace_dir: str | Path | None = None) -> "CodexBackend":
        """Discard episode context explicitly; never reuse another episode's thread."""
        if self._runtime is not None and self.thread_id is not None:
            self._release_shared_thread()
        if trace_dir is not None:
            self.close()
            self.trace_dir = Path(trace_dir).resolve()
        self.thread_id = None
        self.last_turn = {}
        self._pending.clear()
        try:
            if self.process is None:
                self._launch()
            self._check_auth()
            result = self._request("thread/start", {
                "model": self.model,
                "cwd": str(self.trace_dir),
                "approvalPolicy": "never",
                "sandbox": "read-only",
                "baseInstructions": instructions,
                # Keep native context during the episode without filling the user's chat list.
                "ephemeral": True,
                "serviceName": "astra-robomemarena",
                "config": self._overrides,
                "environments": [],
                "runtimeWorkspaceRoots": [],
                "selectedCapabilityRoots": [],
                "dynamicTools": [],
            })
            self.thread_id = result["thread"]["id"]
            if result.get("model", self.model) != self.model:
                raise CodexBackendError("Codex selected a different model")
            self._log("episode", {"thread_id": self.thread_id, "model": self.model,
                                  "reasoning_effort": self.reasoning_effort, "auth": self.auth})
        except BaseException:
            self.close()
            raise
        return self

    def _launch(self) -> None:
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._trace = (self.trace_dir / "codex_protocol.jsonl").open("a", buffering=1)
        if self._runtime is not None:
            self._runtime._attach(self)
            self._log("startup", {"binary": self.codex_bin, "version": self.version,
                                  "shared_runtime_pid": self.process.pid,
                                  "shared_trace": str(self._runtime.trace_dir)})
            return
        self._stderr = (self.trace_dir / "codex_stderr.log").open("a", buffering=1)
        env = dict(os.environ)
        for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "CODEX_BASE_URL", "CODEX_ACCESS_TOKEN"):
            env.pop(key, None)
        self.version = subprocess.run([self.codex_bin, "--version"], env=env, check=True,
                                      capture_output=True, text=True, timeout=15).stdout.strip()
        command = [self.codex_bin]
        for key, value in self._overrides.items():
            command.extend(["-c", f"{key}={json.dumps(value)}"])
        command.extend(["app-server", "--listen", "stdio://"])
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self._stderr, text=True, bufsize=1,
                                        env=env, cwd=self.trace_dir, start_new_session=True)
        self._events = queue.Queue()
        threading.Thread(target=self._reader, args=(self.process.stdout, self._events), daemon=True).start()
        self._log("startup", {"binary": self.codex_bin, "version": self.version})
        self._request("initialize", {
            "clientInfo": {"name": "astra-robomemarena", "title": "Astra RoboMemArena", "version": "0.1"},
            "capabilities": {"experimentalApi": True},
        })
        self._send({"method": "initialized", "params": {}})
        config = self._request("config/read", {"includeLayers": False}).get("config", {})
        if (config.get("model_providers") or {}).get("openai"):
            raise CodexBackendError("Custom OpenAI provider configuration is not allowed; use native ChatGPT login")
        for name in config.get("mcp_servers") or {}:
            self._overrides[f"mcp_servers.{name}.enabled"] = False

    def _check_auth(self) -> None:
        result = self._request("account/read", {"refreshToken": False})
        account = result.get("account") or {}
        self.auth = {"type": account.get("type"), "plan_type": account.get("planType")}
        if account.get("type") != "chatgpt" or not result.get("requiresOpenaiAuth"):
            raise CodexBackendError(f"ChatGPT login required; account type is {account.get('type')!r}. "
                                    "Log in with this Codex binary as the same Unix user. No API fallback.")

    def ask(
        self,
        text: str = "",
        image_paths: Iterable[str | Path] = (),
        *,
        content: list[dict[str, Any]] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> str:
        """Send new observations; ``content`` preserves interleaved frame labels/images."""
        if self.process is None or self.thread_id is None:
            raise CodexBackendError("No active episode; call start or reset first")
        inputs = list(content or [])
        if text:
            inputs.append({"type": "text", "text": text})
        inputs.extend({"type": "localImage", "path": str(path)} for path in image_paths)
        normalized = []
        for item in inputs:
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                normalized.append({"type": "text", "text": item["text"]})
            elif item.get("type") == "localImage":
                path = Path(item["path"]).resolve(strict=True)
                if not path.is_file():
                    raise ValueError(f"Image is not a file: {path}")
                normalized.append({"type": "localImage", "path": str(path)})
            else:
                raise ValueError("Inputs must be text or localImage items")
        if not normalized:
            raise ValueError("A turn requires text or images")
        deadline = time.monotonic() + self.timeout_s
        self.last_turn = {"thread_id": self.thread_id, "status": "starting"}
        try:
            self._check_auth()
            params = {"threadId": self.thread_id, "input": normalized,
                      "model": self.model, "effort": self.reasoning_effort}
            if output_schema is not None:
                params["outputSchema"] = output_schema
            result = self._request("turn/start", params, deadline)
            turn_id = result["turn"]["id"]
            self.last_turn.update(turn_id=turn_id, status="inProgress")
            answer = None
            while True:
                event = self._pending.popleft() if self._pending else self._receive(deadline)
                self._check_event(event)
                data = event.get("params", {})
                if data.get("threadId") != self.thread_id:
                    continue
                if data.get("turnId", turn_id) != turn_id:
                    continue
                if event.get("method") == "thread/tokenUsage/updated":
                    self.last_turn["usage"] = data.get("tokenUsage")
                if event.get("method") == "item/completed":
                    item = data.get("item", {})
                    if item.get("type") == "agentMessage" and item.get("phase") in (None, "final_answer"):
                        answer = item.get("text", "")
                if event.get("method") == "turn/completed":
                    turn = data.get("turn", {})
                    if turn.get("id") != turn_id:
                        continue
                    self.last_turn["status"] = turn.get("status")
                    if turn.get("status") != "completed":
                        raise CodexBackendError(f"Codex turn {turn_id} {turn.get('status')}: {turn.get('error')}")
                    if answer is None:
                        for item in turn.get("items", []):
                            if item.get("type") == "agentMessage" and item.get("phase") in (None, "final_answer"):
                                answer = item.get("text", "")
                    if not isinstance(answer, str) or not answer.strip():
                        raise CodexBackendError("Codex completed without a final text answer")
                    self._log("answer", {**self.last_turn, "text": answer})
                    return answer
        except BaseException as exc:
            self.last_turn.update(status="error", error=f"{type(exc).__name__}: {exc}")
            self._log("failure", self.last_turn)
            self.close()
            raise

    @staticmethod
    def _reader(stream, events: queue.Queue) -> None:
        try:
            for line in stream:
                events.put(json.loads(line))
        except Exception as exc:
            events.put(CodexBackendError(f"Invalid app-server stream: {exc}"))
        finally:
            events.put(CodexBackendError("Codex app-server disconnected; turn was not replayed"))

    def _send(self, message: dict[str, Any]) -> None:
        self._log("send", message)
        if self._runtime is not None:
            self._runtime._send(message)
            return
        try:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise CodexBackendError("Codex app-server disconnected; turn was not replayed") from exc

    def _receive(self, deadline: float) -> dict[str, Any]:
        try:
            event = self._events.get(timeout=max(0, deadline - time.monotonic()))
        except queue.Empty as exc:
            raise TimeoutError(f"Codex timed out after {self.timeout_s}s; turn was not replayed") from exc
        if isinstance(event, Exception):
            raise event
        method = self._requests.get(event.get("id"))
        logged = event
        if method == "config/read" and "result" in event:
            logged = {"id": event["id"], "result": "configuration omitted"}
        elif method == "account/read" and "result" in event:
            account = event["result"].get("account") or {}
            logged = {"id": event["id"], "result": {"type": account.get("type"), "planType": account.get("planType")}}
        self._log("receive", logged)
        return event

    def _check_event(self, event: dict[str, Any]) -> None:
        if "method" in event and "id" in event:
            self._send({"id": event["id"], "error": {"code": -32601, "message": "Evaluation backend does not execute tools or handle login"}})
            raise CodexBackendError(f"Unexpected server request: {event['method']}")
        if event.get("method") == "account/updated" and event.get("params", {}).get("authMode") != "chatgpt":
            raise CodexBackendError("ChatGPT authentication changed during the episode")

    def _request(self, method: str, params: dict[str, Any], deadline: float | None = None) -> dict[str, Any]:
        if self._runtime is not None:
            request_id = self._runtime._reserve(self, method)
        else:
            self._next_id += 1
            request_id = self._next_id
        self._requests[request_id] = method
        self._send({"id": request_id, "method": method, "params": params})
        deadline = deadline if deadline is not None else time.monotonic() + min(30, self.timeout_s)
        while True:
            event = self._receive(deadline)
            self._check_event(event)
            if event.get("id") == request_id:
                if "error" in event:
                    raise CodexBackendError(f"{method} failed: {event['error']}")
                return event["result"]
            self._pending.append(event)

    def _log(self, direction: str, message: dict[str, Any]) -> None:
        with self._log_lock:
            if self._trace is not None:
                self._trace.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(),
                                              "direction": direction, "message": message}, ensure_ascii=False) + "\n")

    def _release_shared_thread(self) -> None:
        thread_id = self.thread_id
        if thread_id is None:
            return
        try:
            if self._runtime._failure is None:
                if self.last_turn.get("turn_id") and self.last_turn.get("status") != "completed":
                    self._request("turn/interrupt", {"threadId": thread_id,
                                  "turnId": self.last_turn["turn_id"]}, time.monotonic() + 5)
                self._request("thread/unsubscribe", {"threadId": thread_id}, time.monotonic() + 5)
        except Exception as exc:
            self._log("close_warning", {"thread_id": thread_id, "error": str(exc)})
        finally:
            self._runtime._unbind(self, thread_id)
            self.thread_id = None

    def close(self) -> None:
        """Stop only this backend's child process; credentials are left untouched."""
        if self._runtime is not None:
            self._release_shared_thread()
            self._runtime._detach(self)
            self.process = None
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
        with self._log_lock:
            for stream in (self._stderr, self._trace):
                if stream is not None:
                    stream.close()
            self._stderr = self._trace = None


class SharedCodexRuntime:
    """One authenticated app-server; clients own disjoint native conversations.

    Start once on the login host, then give each concurrent episode its own
    ``runtime.backend(trace_dir)``. A global transport failure fails all clients;
    closing an individual client never terminates another episode's runtime.
    """

    def __init__(self, trace_dir: str | Path, *, codex_bin: str | None = None,
                 model: str = "gpt-6-astra", reasoning_effort: str = "medium",
                 timeout_s: float = 600):
        self.trace_dir = Path(trace_dir).resolve()
        self._owner = CodexBackend(self.trace_dir, codex_bin=codex_bin, model=model,
                                   reasoning_effort=reasoning_effort, timeout_s=timeout_s)
        self._lock = threading.RLock()
        self._clients: set[CodexBackend] = set()
        self._request_owners: dict[int, CodexBackend] = {}
        self._thread_owners: dict[str, CodexBackend] = {}
        self._early: dict[str, list[dict[str, Any]]] = {}
        self._failure: Exception | None = None
        self._router: threading.Thread | None = None

    def start(self) -> SharedCodexRuntime:
        with self._lock:
            if self._failure is not None:
                raise self._failure
            if self._router is not None:
                return self
            try:
                self._owner._launch()
                self._owner._check_auth()
            except BaseException:
                self._owner.close()
                raise
            self._router = threading.Thread(target=self._route, daemon=True,
                                            name="codex-shared-router")
            deadline = time.monotonic() + 30
            while True:
                try:
                    self._router.start()
                    break
                except RuntimeError as exc:
                    if str(exc) != "can't start new thread" or time.monotonic() >= deadline:
                        raise
                    self._owner._log('startup_resource_wait', {'error': str(exc)})
                    time.sleep(1)
        return self

    def backend(self, trace_dir: str | Path, *, timeout_s: float = 600) -> CodexBackend:
        return CodexBackend(trace_dir, codex_bin=self._owner.codex_bin, model=self._owner.model,
                            reasoning_effort=self._owner.reasoning_effort,
                            timeout_s=timeout_s, runtime=self)

    def _attach(self, client: CodexBackend) -> None:
        with self._lock:
            if self._failure is not None:
                raise self._failure
            if self._router is None or self._owner.process is None:
                raise CodexBackendError("Start the shared runtime before its clients")
            client.process, client.version = self._owner.process, self._owner.version
            client._overrides = dict(self._owner._overrides)
            client._events = queue.Queue()
            self._clients.add(client)

    def _reserve(self, client: CodexBackend, method: str) -> int:
        with self._lock:
            if self._failure is not None:
                raise self._failure
            self._owner._next_id += 1
            request_id = self._owner._next_id
            self._owner._requests[request_id] = method
            self._request_owners[request_id] = client
            return request_id

    def _send(self, message: dict[str, Any]) -> None:
        with self._lock:
            if self._failure is not None:
                raise self._failure
            self._owner._send(message)

    def _unbind(self, client: CodexBackend, thread_id: str) -> None:
        with self._lock:
            if self._thread_owners.get(thread_id) is client:
                del self._thread_owners[thread_id]
            self._early.pop(thread_id, None)

    def _detach(self, client: CodexBackend) -> None:
        with self._lock:
            self._clients.discard(client)
            for thread_id, owner in list(self._thread_owners.items()):
                if owner is client:
                    del self._thread_owners[thread_id]

    def _route(self) -> None:
        try:
            while self._failure is None:
                event = self._owner._receive(time.monotonic() + 86400)
                data = event.get("params", {})
                thread_id = data.get("threadId") or (data.get("thread") or {}).get("id")
                with self._lock:
                    if "id" in event and "method" not in event:
                        client = self._request_owners.pop(event["id"], None)
                        if client is None:
                            if "error" in event:
                                raise CodexBackendError(f"Unowned app-server error: {event['error']}")
                            continue
                        if client not in self._clients:
                            continue
                        method = self._owner._requests[event["id"]]
                        new_thread = (event.get("result", {}).get("thread") or {}).get("id")
                        if method == "thread/start" and new_thread:
                            if new_thread in self._thread_owners:
                                raise CodexBackendError("App-server reused an existing episode thread")
                            self._thread_owners[new_thread] = client
                        client._events.put(event)
                        for pending in self._early.pop(new_thread, []):
                            client._events.put(pending)
                    elif thread_id:
                        client = self._thread_owners.get(thread_id)
                        if client is not None:
                            client._events.put(event)
                        elif any(self._owner._requests[key] == "thread/start" for key in self._request_owners):
                            self._early.setdefault(thread_id, []).append(event)
                    elif event.get("method") in {"error", "fatal"} or "error" in event:
                        raise CodexBackendError(f"Global app-server error: {event}")
                    elif "method" in event and "id" in event:
                        self._owner._send({"id": event["id"], "error": {
                            "code": -32601, "message": "Shared evaluation runtime cannot handle this request"}})
                        raise CodexBackendError(f"Unscoped server request: {event['method']}")
                    else:
                        if event.get("method") == "account/updated" and data.get("authMode") != "chatgpt":
                            raise CodexBackendError("Shared ChatGPT authentication changed")
                        for client in self._clients:
                            client._events.put(event)
        except Exception as exc:
            self._fail(exc)

    def _fail(self, error: Exception) -> None:
        with self._lock:
            if self._failure is None:
                self._failure = error
                self._owner._log("runtime_failure", {"error": str(error)})
                for client in self._clients:
                    client._events.put(error)
        self._owner.close()

    def close(self) -> None:
        self._fail(CodexBackendError("Shared app-server runtime closed"))
        if self._router is not None and self._router.ident is not None and self._router is not threading.current_thread():
            self._router.join(timeout=5)

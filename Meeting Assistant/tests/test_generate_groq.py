"""Groq provider: selection, request shape, and the real network code against a local server that speaks
Groq's (OpenAI-style) streaming protocol. api.groq.com itself was not reachable where this was built."""
import http.server
import json
import threading
import time

import pytest

import generate as g
from followup import Cue

THR = 0.78


def req(q="what is your biggest weakness?"):
    return g.build_request(q, [g.Match(g.Entry(q, "ANSWER " + q, bank="t"), 0.9)], thr=THR)


def chunk(text=None, finish=None, usage=None, empty=False):
    c = {"id": "x", "object": "chat.completion.chunk", "choices": [] if empty else
         [{"index": 0, "delta": ({"content": text} if text is not None else {}), "finish_reason": finish}]}
    if usage:
        c["usage"] = usage
    return c


def sse(*chunks, done=True):
    out = [f"data: {json.dumps(c)}\n\n" if not isinstance(c, str) else c for c in chunks]
    if done:
        out.append("data: [DONE]\n\n")
    return out


def run(gen, request=None):
    events, done = [], threading.Event()
    gen.on_event = lambda j, k, d: (events.append((k, d)), done.set() if k in ("done", "error") else None)
    gen.start(request or req())
    assert done.wait(10), events
    return events


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    for var in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(g, "GROQ_KEY_FILE", tmp_path / "groq_key.txt")
    monkeypatch.setattr(g, "KEY_FILE", tmp_path / "anthropic_key.txt")
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")


# ---------- which provider, which model, what the status says ----------

def test_auto_prefers_groq_when_there_is_a_groq_key(monkeypatch, tmp_path):
    gen = g.Generator({"generate": True}, lambda *a: None)
    assert gen.provider == "anthropic"                                   # nothing configured: the old default
    monkeypatch.setenv("GROQ_API_KEY", "gsk_env")
    assert gen.provider == "groq" and gen.model == g.GROQ_DEFAULT_MODEL
    monkeypatch.delenv("GROQ_API_KEY")
    (tmp_path / "groq_key.txt").write_text("# from the console\ngsk_file\n", encoding="utf-8")
    assert g.groq_key() == "gsk_file" and gen.provider == "groq"


def test_explicit_provider_and_model_rules(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    assert g.Generator({"generate_provider": "anthropic"}, lambda *a: None).provider == "anthropic"
    gen = g.Generator({"generate_provider": "groq", "generate_model": "openai/gpt-oss-120b"}, lambda *a: None)
    assert gen.provider == "groq" and gen.model == "openai/gpt-oss-120b"
    old_default = g.Generator({"generate_provider": "groq", "generate_model": "claude-opus-5-5"}, lambda *a: None)
    assert old_default.model == g.GROQ_DEFAULT_MODEL                      # a Claude model name can't be sent to Groq
    assert g.Generator({"generate_provider": "anthropic"}, lambda *a: None).model == "claude-opus-5-5"
    assert g.Generator({"generate_provider": "anthropic", "generate_model": "claude-sonnet-5-5"}, lambda *a: None).model == "claude-sonnet-5-5"


def test_status_messages(monkeypatch):
    mk = lambda **c: g.Generator({"generate": True, **c}, lambda *a: None)
    ok, why = mk().status()
    assert not ok and "groq_key.txt" in why and "anthropic_key.txt" in why           # auto, no keys: both ways are explained
    ok, why = mk(generate_provider="groq").status()
    assert not ok and "No Groq key found" in why
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    assert mk().status() == (True, "")
    assert mk(generate_provider="groq").status() == (True, "")
    assert mk(generate=False).status() == (False, "Generated answers are switched off.")
    monkeypatch.delenv("GROQ_API_KEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    monkeypatch.setitem(__import__("sys").modules, "anthropic", None)
    ok, why = mk().status()
    assert not ok and "pip install anthropic" in why


# ---------- request and stream parsing, through the test hook ----------

def hook(*chunks, **kw):
    seen = []

    def http(payload):
        seen.append(payload)
        yield from [ln.strip()[5:].strip() for ln in sse(*chunks, **kw) if ln.startswith("data:")]
    http.seen = seen
    return http


def test_payload_is_what_groq_expects():
    h = hook(chunk("hi"))
    gen = g.Generator({"generate": True, "generate_max_tokens": 900}, lambda *a: None, http=h)
    run(gen)
    p = h.seen[0]
    assert p["model"] == g.GROQ_DEFAULT_MODEL and p["stream"] is True and p["max_tokens"] == 900
    assert p["stream_options"] == {"include_usage": True} and p["temperature"] == 0.6
    assert [m["role"] for m in p["messages"]] == ["system", "user"]
    assert "Never use bullet points" in p["messages"][0]["content"]
    assert "<question>\nwhat is your biggest weakness?\n</question>" in p["messages"][1]["content"]
    assert "cache_control" not in json.dumps(p) and "output_config" not in p      # Anthropic-only fields stay out


def test_profile_is_part_of_the_system_message():
    h = hook(chunk("hi"))
    gen = g.Generator({"generate": True}, lambda *a: None, http=h)
    run(gen, g.build_request("q?", [], thr=THR, profile="I have 7 years in QA."))
    sysmsg = h.seen[0]["messages"][0]["content"]
    assert "PROFILE (facts about the user):\nI have 7 years in QA." in sysmsg
    assert sysmsg.index("Never write placeholders") < sysmsg.index("PROFILE (facts about the user)")      # the facts follow the rules


def test_streams_deltas_usage_and_model():
    h = hook(chunk("Own it. "), chunk("I work "), chunk("hard."), chunk(finish="stop"),
             chunk(empty=True, usage={"prompt_tokens": 812, "completion_tokens": 44,
                                      "prompt_tokens_details": {"cached_tokens": 700}}))
    ev = run(g.Generator({"generate": True}, lambda *a: None, http=h))
    assert [k for k, _ in ev] == ["start", "delta", "delta", "delta", "done"]
    done = ev[-1][1]
    assert done["text"] == "Own it. I work hard." and done["provider"] == "groq" and done["model"] == g.GROQ_DEFAULT_MODEL
    assert (done["input_tokens"], done["output_tokens"], done["cached_tokens"]) == (812, 44, 700) and not done["truncated"]


def test_usage_in_the_x_groq_field_and_length_truncation():
    h = hook(chunk("cut"), {"choices": [{"delta": {}, "finish_reason": "length"}], "x_groq": {"usage": {"prompt_tokens": 5, "completion_tokens": 9}}})
    done = run(g.Generator({"generate": True}, lambda *a: None, http=h))[-1][1]
    assert done["truncated"] is True and done["input_tokens"] == 5 and done["output_tokens"] == 9


def test_leading_blank_output_and_garbage_lines_are_ignored():
    h = hook(chunk("\n\n"), chunk("  Real "), "data: not json at all\n\n", chunk("answer."), done=False)
    done = run(g.Generator({"generate": True}, lambda *a: None, http=h))[-1][1]
    assert done["text"] == "Real answer."


def test_thinking_tags_never_reach_the_screen():
    h = hook(chunk("<thi"), chunk("nk>weighing it up</th"), chunk("ink>Plan. "), chunk("Do it."))
    ev = run(g.Generator({"generate": True}, lambda *a: None, http=h))
    assert ev[-1][1]["text"] == "Plan. Do it." and not any("think" in str(d) for k, d in ev if k == "delta")


@pytest.mark.parametrize("pieces,expect", [
    (["a<think>x</think>b"], "ab"),
    (["<think>never closed"], ""),
    (["x <th", "at> y"], "x <that> y"),                       # an angle bracket that isn't a think tag survives
    (["<think>a</think>", "<think>b</think>c"], "c"),
    (["plain"], "plain"),
])
def test_think_filter_cases(pieces, expect):
    f = g._ThinkFilter()
    assert "".join(f.feed(p) for p in pieces) + f.flush() == expect


# ---------- the real HTTP code, against a local server ----------

class FakeGroq:
    """Speaks the same wire protocol as api.groq.com for the parts this app uses."""

    def __init__(self):
        self.requests = []          # (method, path, headers dict, parsed body)
        self.script = []            # per POST: dict(status, events, delay, headers, body)
        self.models = {"data": []}
        self.broken = threading.Event()
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _body(self):
                n = int(self.headers.get("content-length") or 0)
                return json.loads(self.rfile.read(n) or b"{}") if n else {}

            def do_GET(self):
                outer.requests.append(("GET", self.path, dict(self.headers), None))
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(outer.models).encode())

            def do_POST(self):
                body = self._body()
                outer.requests.append(("POST", self.path, dict(self.headers), body))
                step = outer.script.pop(0) if outer.script else {"status": 200, "events": sse(chunk("ok"))}
                self.send_response(step.get("status", 200))
                for k, v in step.get("headers", {}).items():
                    self.send_header(k, v)
                if step.get("status", 200) != 200:
                    self.send_header("content-type", step.get("ctype", "application/json"))
                    self.end_headers()
                    self.wfile.write((step.get("body") or json.dumps({"error": {"message": "nope"}})).encode())
                    return
                self.send_header("content-type", "text/event-stream")
                self.end_headers()
                try:
                    for piece in step["events"]:
                        self.wfile.write(piece.encode())
                        self.wfile.flush()
                        time.sleep(step.get("delay", 0))
                except (BrokenPipeError, ConnectionResetError):
                    outer.broken.set()

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/openai/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def groq(monkeypatch):
    srv = FakeGroq()
    monkeypatch.setenv("GROQ_API_KEY", "gsk_testkey")
    yield srv
    srv.close()


def gen_for(srv, **cfg):
    return g.Generator({"generate": True, "generate_provider": "groq", "groq_base_url": srv.url, **cfg}, lambda *a: None)


def test_real_request_headers_and_body(groq):
    groq.script = [{"events": sse(chunk("Hello "), chunk("there."), chunk(finish="stop"),
                                  chunk(empty=True, usage={"prompt_tokens": 10, "completion_tokens": 3}))}]
    ev = run(gen_for(groq))
    assert ev[-1][0] == "done" and ev[-1][1]["text"] == "Hello there."
    method, path, headers, body = groq.requests[0]
    assert (method, path) == ("POST", "/openai/v1/chat/completions")
    low = {k.lower(): v for k, v in headers.items()}
    assert low["authorization"] == "Bearer gsk_testkey"
    assert low["user-agent"] == g.USER_AGENT and "python" not in low["user-agent"].lower()      # Cloudflare blocks Python's default
    assert low["content-type"] == "application/json" and low["accept"] == "text/event-stream"
    assert body["model"] == g.GROQ_DEFAULT_MODEL and body["messages"][0]["role"] == "system"
    assert "gsk_testkey" not in path and "gsk_testkey" not in json.dumps(body)                  # the key only travels in the header


def test_text_arrives_while_the_server_is_still_talking(groq):
    groq.script = [{"events": sse(*[chunk(f"w{i} ") for i in range(8)]), "delay": 0.1}]
    gen = gen_for(groq)
    stamps, done = [], threading.Event()
    gen.on_event = lambda j, k, d: (stamps.append((k, time.monotonic())), done.set() if k == "done" else None)
    t0 = time.monotonic()
    gen.start(req())
    assert done.wait(10)
    first = next(t for k, t in stamps if k == "delta") - t0
    last = next(t for k, t in stamps if k == "done") - t0
    assert first < 0.35 and last > 0.7                       # first words well before the stream ends


def test_a_json_line_split_across_packets_is_reassembled(groq):
    line = f"data: {json.dumps(chunk('whole'))}\n\n"
    groq.script = [{"events": [line[:15], line[15:], "data: [DONE]\n\n"], "delay": 0.05}]
    assert run(gen_for(groq))[-1][1]["text"] == "whole"


def test_stream_that_just_closes_without_done_still_finishes(groq):
    groq.script = [{"events": sse(chunk("abc"), done=False)}]
    assert run(gen_for(groq))[-1][1]["text"] == "abc"


def test_a_newer_question_closes_the_connection(groq):
    groq.script = [{"events": sse(*[chunk(f"w{i} ") for i in range(200)]), "delay": 0.03}]
    gen = gen_for(groq)
    seen = []
    gen.on_event = lambda j, k, d: seen.append((j, k))
    gen.start(req("first question here please?"))
    time.sleep(0.4)
    gen.cancel()
    assert groq.broken.wait(5)                               # the server saw the client hang up: no tokens wasted
    n = len(seen)
    time.sleep(0.3)
    assert len(seen) == n and not any(k in ("done", "error") for _, k in seen)


def test_401_is_explained_and_stops_further_requests(groq):
    groq.script = [{"status": 401, "body": json.dumps({"error": {"message": "Invalid API Key", "type": "invalid_request_error", "code": "invalid_api_key"}})}]
    gen = gen_for(groq)
    ev = run(gen)
    assert ev[-1][0] == "error" and "Groq key was rejected" in ev[-1][1] and gen.disabled_reason
    n = len(groq.requests)
    ev = run(gen)
    assert "rejected" in ev[-1][1] and len(groq.requests) == n          # latched: no second request


def test_429_says_how_long_to_wait_and_is_not_fatal(groq):
    groq.script = [{"status": 429, "headers": {"Retry-After": "7"}, "body": json.dumps({"error": {"message": "Rate limit reached"}})}]
    gen = gen_for(groq)
    msg = run(gen)[-1][1]
    assert "rate limit" in msg.lower() and "7s" in msg and not gen.disabled_reason
    groq.script = [{"events": sse(chunk("back"))}]
    assert run(gen)[-1][1]["text"] == "back"                            # and it recovers on its own


def test_a_retired_default_model_heals_itself(groq):
    groq.models = {"data": [
        {"id": "whisper-large-v3", "active": True, "context_window": 448},
        {"id": "llama-guard-4-12b", "active": True, "context_window": 131072},
        {"id": "openai/gpt-oss-120b", "active": True, "context_window": 131072},
        {"id": "some-old-model", "active": False, "context_window": 999999},
    ]}
    groq.script = [{"status": 400, "body": json.dumps({"error": {"message": "The model `llama-3.3-70b-versatile` has been decommissioned", "code": "model_decommissioned"}})},
                   {"events": sse(chunk("works now"))}]
    gen = gen_for(groq)
    ev = run(gen)
    assert ev[-1][0] == "done" and ev[-1][1]["text"] == "works now" and ev[-1][1]["model"] == "openai/gpt-oss-120b"
    posts = [r for r in groq.requests if r[0] == "POST"]
    assert [p[3]["model"] for p in posts] == ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"]
    assert gen.model == "openai/gpt-oss-120b" and not gen.disabled_reason


def test_a_model_the_user_chose_is_not_silently_swapped(groq):
    groq.script = [{"status": 404, "body": json.dumps({"error": {"message": "model does not exist", "code": "model_not_found"}})}]
    gen = gen_for(groq, generate_model="my-own-model")
    msg = run(gen)[-1][1]
    assert "it generate --models" in msg and gen.disabled_reason
    assert not any(r[0] == "GET" for r in groq.requests)


def test_server_trouble_and_html_error_pages(groq):
    groq.script = [{"status": 503, "body": json.dumps({"error": {"message": "overloaded"}})}]
    assert "having trouble (503)" in run(gen_for(groq))[-1][1]
    groq.script = [{"status": 403, "ctype": "text/html", "body": "<html>Access denied | Cloudflare</html>"}]
    gen = gen_for(groq)
    assert "403" in run(gen)[-1][1] and not gen.disabled_reason


def test_an_error_object_in_the_middle_of_a_stream(groq):
    groq.script = [{"events": sse(chunk("part "), {"error": {"message": "server exploded", "status": 500}}, done=False)}]
    ev = run(gen_for(groq))
    assert ev[-1][0] == "error" and "having trouble (500)" in ev[-1][1]


def test_unreachable_server_is_just_offline(groq):
    gen = gen_for(groq)
    gen.cfg["groq_base_url"] = "http://127.0.0.1:1/openai/v1"           # nothing listens there
    ev = run(gen)
    assert ev[-1][0] == "error" and "Couldn't reach the API" in ev[-1][1] and not gen.disabled_reason


def test_list_models_hides_audio_and_guard_models_and_retired_ones(groq):
    groq.models = {"data": [{"id": "llama-3.3-70b-versatile", "context_window": 131072},
                            {"id": "whisper-large-v3-turbo", "context_window": 448},
                            {"id": "meta-llama/llama-guard-4-12b", "context_window": 131072},
                            {"id": "playai-tts", "context_window": 8192},
                            {"id": "old", "active": False, "context_window": 1},
                            {"id": "openai/gpt-oss-20b", "context_window": 131072}]}
    assert gen_for(groq).list_models() == [("llama-3.3-70b-versatile", 131072), ("openai/gpt-oss-20b", 131072)]
    low = {k.lower(): v for k, v in groq.requests[0][2].items()}
    assert groq.requests[0][1] == "/openai/v1/models" and low["authorization"] == "Bearer gsk_testkey"


def test_end_to_end_request_survives_the_overlay_prompt():
    """The prompt built for the overlay goes through the Groq path unchanged in meaning."""
    h = hook(chunk("ok"))
    gen = g.Generator({"generate": True}, lambda *a: None, http=h)
    parent = g.Entry("Tell me about a time you led a team", "I led {{N}} people.", bank="t")
    r = g.build_request("what was the outcome?", [], thr=THR, parent=parent, cue=Cue("outcome", 1.0, "x"))
    run(gen, r)
    assert "I led {{N}} people." in h.seen[0]["messages"][1]["content"]


@pytest.mark.parametrize("raw,clean", [
    ("Here's an answer:\n\n**Own it.** I did.", "Own it. I did."),
    ("- I led six people\n- We shipped on time\n\nI learned a lot.", "I led six people. We shipped on time.\n\nI learned a lot."),
    ("1. Plan it\n2) Do it", "Plan it. Do it."),
    ("## My answer\nI led {{team size}} people.", "My answer\nI led team size people."),
    ("A bullet-free answer with a dash - in the middle.", "A bullet-free answer with a dash - in the middle."),
    ("Sure! Here is a draft:\nI led the team.", "I led the team."),
    ('"I led the team and learned a lot."', "I led the team and learned a lot."),
    ("\u201cI led the team.\u201d", "I led the team."),
    ('"First." and then "second."', '"First." and then "second."'),          # inner quotes: not a wrapper, leave alone
    ("Here at the company we ship weekly.", "Here at the company we ship weekly."),   # not chatter
    ("Here is why I care:\nBecause it matters.", "Because it matters."),
    ("Plain answer.", "Plain answer."),
    ("Here's an answer:", "Here's an answer:"),                            # nothing after it: don't blank the draft
])
def test_tidy_draft(raw, clean):
    assert g.tidy_draft(raw) == clean


def test_the_finished_draft_is_tidied_but_streaming_is_untouched():
    h = hook(chunk("Here's an answer:\n"), chunk("**Own it.** "), chunk("Done."))
    ev = run(g.Generator({"generate": True}, lambda *a: None, http=h))
    assert ev[-1][1]["text"] == "Own it. Done."
    assert "".join(d for k, d in ev if k == "delta").startswith("Here's an answer:")      # the screen swaps in the clean text at the end

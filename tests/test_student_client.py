import dataclasses
import json
import ssl
import urllib.error

from cua.eval.screenspot import predictors
from cua.eval.screenspot.config import EvalConfig
from cua.eval.student.backends import OpenAIChatStudent, StudentError
from cua.rl.policies import OpenAICompatiblePolicy
from cua.student.client import (
    ChatResponse,
    OpenAICompatibleClient,
    RetryPolicy,
    TransportError,
    chat_headers,
    chat_payload,
    chat_url,
    choice_text,
    choice_text_or_empty,
    choice_text_parts,
)

MESSAGES = [
    {"role": "system", "content": "SYS"},
    {"role": "user", "content": [{"type": "text", "text": "hi"}]},
]
ANSWER = json.dumps({"choices": [{"message": {"content": "<think>t</think>\naction: done()"}}]})
OBSERVATION = {"system": "s", "text": "t", "image_b64": "QUJD", "step_index": 0}


class FakeUrllibResponse:
    def __init__(self, text, status=200):
        self.text, self.status = text, status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.text.encode()


def recording_opener(outcomes, log):
    def opener(request, timeout=None):
        log.append(
            {
                "url": request.full_url,
                "headers": dict(request.headers),
                "body": json.loads(request.data),
                "timeout": timeout,
            }
        )
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return opener


def recording_transport(outcomes, log):
    def send(url, payload, headers, timeout_s):
        log.append({"url": url, "headers": headers, "body": payload, "timeout": timeout_s})
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return send


def test_chat_url_accepts_a_base_url_or_the_full_path():
    assert chat_url("http://x/v1") == "http://x/v1/chat/completions"
    assert chat_url("http://x/v1/") == "http://x/v1/chat/completions"
    assert chat_url("http://x/v1/chat/completions") == "http://x/v1/chat/completions"


def test_headers_carry_the_key_only_when_there_is_one_unless_the_caller_always_sends_it():
    assert chat_headers("k") == {"Content-Type": "application/json", "Authorization": "Bearer k"}
    assert chat_headers("") == {"Content-Type": "application/json"}
    assert chat_headers("", send_empty_auth=True) == {"Content-Type": "application/json", "Authorization": "Bearer "}


def test_payload_is_one_shape_for_every_caller():
    assert chat_payload("m", MESSAGES, 0.0, 512) == {
        "model": "m",
        "messages": MESSAGES,
        "temperature": 0.0,
        "max_tokens": 512,
    }


def test_content_readings_differ_only_where_the_callers_do():
    assert choice_text(ANSWER) == "<think>t</think>\naction: done()"
    assert choice_text('{"choices": [{"message": {"content": ""}}]}') == ""
    assert choice_text_or_empty('{"choices": [{"message": {}}]}') == ""
    assert (
        choice_text_parts('{"choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}, {"x": 1}]}}]}') == "ab"
    )
    for reading in (choice_text, choice_text_parts):
        try:
            reading('{"choices": [{"message": {}}]}')
            raise AssertionError("a completion with no content must raise for a strict reading")
        except KeyError:
            pass


def test_rl_policy_sends_the_recorded_request_and_returns_the_answer():
    log = []
    policy = OpenAICompatiblePolicy(
        "http://x/v1", "m", api_key="k", opener=recording_opener([FakeUrllibResponse(ANSWER)], log)
    )
    assert policy.act(OBSERVATION) == "<think>t</think>\naction: done()"
    assert log == [
        {
            "url": "http://x/v1/chat/completions",
            "headers": {"Content-type": "application/json", "Authorization": "Bearer k"},
            "body": {
                "model": "m",
                "messages": [
                    {"role": "system", "content": "s"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
                            {"type": "text", "text": "t"},
                        ],
                    },
                ],
                "temperature": 0.0,
                "max_tokens": 512,
            },
            "timeout": 180,
        }
    ]


def test_rl_policy_retries_transient_and_5xx_but_never_a_4xx():
    slept = []
    ticks = iter(range(0, 100000, 7))
    policy = OpenAICompatiblePolicy(
        "http://x/v1",
        "m",
        opener=recording_opener([ssl.SSLEOFError("EOF"), FakeUrllibResponse(ANSWER)], []),
        backoff_s=3,
        sleep=slept.append,
        wait_s=20,
        clock=lambda: next(ticks),
    )
    assert policy.act(OBSERVATION).endswith("done()") and slept == [3]
    fatal = OpenAICompatiblePolicy(
        "http://x/v1", "m", opener=recording_opener([urllib.error.HTTPError("http://x", 404, "no", {}, None)], [])
    )
    try:
        fatal.act(OBSERVATION)
        raise AssertionError("a 4xx must reach the caller unchanged")
    except urllib.error.HTTPError as e:
        assert e.code == 404


def test_a_zero_second_wait_budget_makes_one_request_and_never_sleeps():
    log, slept = [], []
    policy = OpenAICompatiblePolicy(
        "http://x/v1",
        "m",
        opener=recording_opener([TimeoutError("timed out")], log),
        sleep=slept.append,
        wait_s=0,
        clock=lambda: 0.0,
    )
    try:
        policy.act(OBSERVATION)
        raise AssertionError("a zero-second budget must be spent by the first failure")
    except ConnectionError as e:
        assert str(e) == "policy endpoint unreachable after 1 attempts in 0s: timed out"
    assert len(log) == 1 and slept == []


def test_student_request_shape_and_list_content():
    log = []
    student = OpenAIChatStudent(endpoint="http://s/v1", model="cua-student", api_key="sk", max_tokens=64)
    student.client.transport = recording_transport(
        [ChatResponse(200, '{"choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}]}}]}')], log
    )
    assert student.complete(MESSAGES) == "ab"
    assert log == [
        {
            "url": "http://s/v1/chat/completions",
            "headers": {"Content-Type": "application/json", "Authorization": "Bearer sk"},
            "body": {"model": "cua-student", "messages": MESSAGES, "temperature": 0.0, "max_tokens": 64},
            "timeout": 180,
        }
    ]


def test_student_raises_on_a_refusal_an_unusable_body_and_a_spent_budget():
    slept = []
    student = OpenAIChatStudent(endpoint="http://s/v1", model="m")
    student.client.sleep = slept.append
    student.client.transport = recording_transport([ChatResponse(404, "nope")], [])
    try:
        student.complete(MESSAGES)
        raise AssertionError("a 404 is the endpoint's verdict, not a retry")
    except StudentError as e:
        assert str(e) == "http://s/v1/chat/completions returned HTTP 404: nope"
    student.client.transport = recording_transport([ChatResponse(200, '{"choices": []}')], [])
    try:
        student.complete(MESSAGES)
        raise AssertionError("an unusable body must raise")
    except StudentError as e:
        assert "unexpected response shape" in str(e)
    student.client.transport = recording_transport([ChatResponse(500, "boom")] * 3, [])
    try:
        student.complete(MESSAGES)
        raise AssertionError("a spent retry budget must raise")
    except StudentError as e:
        assert "unreachable after 3 attempts (HTTP 500: boom)" in str(e)
    assert slept == [90, 180]
    student.client.transport = recording_transport([TransportError("dropped")] * 3, [])
    try:
        student.complete(MESSAGES)
        raise AssertionError("a dead endpoint must raise")
    except StudentError as e:
        assert "unreachable after 3 attempts (transport error: dropped)" in str(e)


def test_screenspot_scores_a_dead_endpoint_as_a_parse_failure_instead_of_aborting():
    log, slept = [], []
    settings = predictors.EVAL
    predictors.EVAL = dataclasses.replace(
        settings, screenspot_endpoint="http://g/v1/", screenspot_api_key="", screenspot_workers=1
    )
    try:
        predictor = predictors.OpenAIPredictor("grounder", dataclasses.replace(EvalConfig(), max_tokens=64))
    finally:
        predictors.EVAL = settings
    predictor.client.sleep = slept.append
    predictor.client.transport = recording_transport([ChatResponse(200, ANSWER)], log)
    assert predictor.client.chat(MESSAGES) == "<think>t</think>\naction: done()"
    # The key is always sent, even blank: some endpoints reject a request with no Authorization header at all.
    assert log[0]["headers"] == {"Content-Type": "application/json", "Authorization": "Bearer "}
    assert log[0]["url"] == "http://g/v1/chat/completions"
    assert log[0]["body"]["model"] == "grounder" and log[0]["body"]["max_tokens"] == 64
    predictor.client.transport = recording_transport([urllib.error.URLError("down")] * 3, [])
    assert predictor.client.chat(MESSAGES) == "" and slept == [2, 4]


def test_a_client_with_no_retries_reports_the_first_failure():
    client = OpenAICompatibleClient("http://x/v1", "m", retry=RetryPolicy(attempts=1))
    client.transport = recording_transport([TransportError("gone")], [])
    assert client.chat(MESSAGES) == ""

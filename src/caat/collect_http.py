"""Collect confidence signals from an OpenAI-compatible Chat Completions API."""

import json
from codecs import getincrementaldecoder
from math import exp, isfinite
from random import Random
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def _logprobs_rejected(body):
    try:
        error = json.loads(body)
    except ValueError:
        return False
    if not isinstance(error, dict):
        return False
    error = error.get("error", error)
    if not isinstance(error, dict):
        return False
    if error.get("code") == "unsupported_parameter" and error.get("param") in (
        "logprobs",
        "top_logprobs",
    ):
        return True
    message = str(error.get("message", "")).lower()
    return "logprobs" in message and any(
        reason in message
        for reason in ("not support", "unsupported", "not available", "max allowed")
    )


def _details(choice):
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    return content if isinstance(content, list) else []


def _score(token):
    value = token.get("logprob") if isinstance(token, dict) else None
    # Some APIs use -9999 as a placeholder for unavailable probabilities.
    if (
        type(value) in (int, float)
        and isfinite(value)
        and value <= 0
        and value != -9999
    ):
        return value
    return None


def _visible_prefix(pieces, text, stopped):
    rendered = "".join(pieces)
    if rendered == text:
        return len(pieces)
    if stopped and rendered.startswith(text):
        # Some servers include the stop sequence in the trace but omit it from the text.
        remaining = len(text)
        for count, piece in enumerate(pieces, 1):
            remaining -= len(piece)
            if remaining <= 0:
                return count if remaining == 0 else 0
    return 0


def _visible_tokens(choice, text):
    details = _details(choice)
    if not text or not all(isinstance(token, dict) for token in details):
        return []
    pieces = [token.get("token") for token in details]
    if not all(isinstance(piece, str) for piece in pieces):
        return []
    stopped = choice.get("finish_reason") == "stop"
    length = _visible_prefix(pieces, text, stopped)
    if not length:
        encoded = [token.get("bytes") for token in details]
        if not all(isinstance(piece, list) for piece in encoded):
            return []
        # A UTF-8 character may span more than one token.
        decoder = getincrementaldecoder("utf-8")()
        try:
            pieces = [decoder.decode(bytes(piece)) for piece in encoded]
            decoder.decode(b"", final=True)
        except (TypeError, ValueError):
            return []
        length = _visible_prefix(pieces, text, stopped)
    scores = [_score(token) for token in details[:length]]
    if any(score is None for score in scores):
        return []
    return [
        {"text": piece, "logprob": score}
        for piece, score in zip(pieces[:length], scores, strict=True)
    ]


class ChatCompletionsBackend:
    def __init__(self, base_url, model, *, seed, api_key=None, logprobs=True):
        url = urlsplit(base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username is not None
            or url.query
            or url.fragment
        ):
            raise ValueError("--base-url needs an HTTP(S) API root without credentials")
        if api_key is not None and any(not 33 <= ord(char) <= 126 for char in api_key):
            raise ValueError(
                "the API key environment variable contains invalid characters"
            )
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.seed = seed
        self.random = Random(seed)
        self.api_key = api_key
        self.logprobs = logprobs
        self.top_logprobs = logprobs
        self.metadata = {
            "backend": "chat-completions",
            "base_url": base_url,
            "model": model,
            "seed": seed,
            "http_requests": 0,
        }

    def _complete(self, prompt, max_tokens, *, sampled=False, **options):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7 if sampled else 0,
            "top_p": 0.9 if sampled else 1,
            "seed": self.random.randrange(2**31) if sampled else self.seed,
            **options,
        }
        request = Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        if self.api_key:
            request.add_unredirected_header("Authorization", f"Bearer {self.api_key}")
        self.metadata["http_requests"] += 1
        try:
            with urlopen(request, timeout=120) as response:
                choice = json.load(response)["choices"][0]
        except HTTPError as exc:
            with exc:
                if (
                    options.get("logprobs")
                    and exc.code in {400, 422}
                    and _logprobs_rejected(exc.read(8192))
                ):
                    return None
            raise ValueError(f"Chat Completions returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError) as exc:
            raise ValueError(f"could not reach Chat Completions at {self.url}") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("invalid Chat Completions response") from exc
        if not isinstance(choice, dict):
            raise ValueError("invalid Chat Completions choice")
        return choice

    def generate(self, prompt, max_new_tokens, *, sampled=False, tokens=False):
        options = {"logprobs": True} if tokens and self.logprobs else {}
        choice = self._complete(prompt, max_new_tokens, sampled=sampled, **options)
        if choice is None:
            self.logprobs = False
            choice = self._complete(prompt, max_new_tokens, sampled=sampled)
        message = choice.get("message")
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str):
            raise ValueError("Chat Completions choice needs string message content")
        return text, _visible_tokens(choice, text) if options else []

    def p_true(self, prompt):
        if not self.logprobs or not self.top_logprobs:
            return None
        choice = self._complete(prompt, 1, logprobs=True, top_logprobs=20)
        if choice is None:
            self.top_logprobs = False
            return None
        details = _details(choice)
        if not details or not isinstance(details[0], dict):
            return None
        first = details[0]
        alternatives = first.get("top_logprobs")
        candidates = (
            [first, *alternatives] if isinstance(alternatives, list) else [first]
        )
        scores = {}
        for candidate in candidates:
            score = _score(candidate)
            if score is not None and candidate.get("token") in ("A", "B"):
                scores[candidate["token"]] = score
        if scores.keys() != {"A", "B"}:
            return None
        peak = max(scores.values())
        a, b = exp(scores["A"] - peak), exp(scores["B"] - peak)
        return a / (a + b)

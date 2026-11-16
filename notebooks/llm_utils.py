import math
import time
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MAX_EMAIL_CHARS = 4000
NUM_CTX = 6144

SYSTEM_PROMPT_SPAM_1 = """You are an email security analyst. Decide whether an email is UNWANTED or LEGITIMATE for the person who received it.

UNWANTED means any email the recipient should not have received or should not trust, in particular:
- phishing: messages that try to steal credentials, card data or personal information, e.g. fake account, security, delivery, tax or payment notices that ask the reader to log in, verify, update or confirm details;
- business email compromise (BEC) and impersonation: messages pretending to be an executive, colleague, supplier or partner that push for an unusual action such as a wire transfer, changed bank details, gift cards, invoice payment or sending confidential documents, often with urgency or secrecy;
- scams and fraud: advance-fee schemes, inheritance or lottery winnings, investment or job offers too good to be true, requests for help moving money, romance or charity scams;
- malware lures: unexpected invoices, documents or links the reader is pushed to open;
- spam: unsolicited bulk or commercial advertising, e.g. pharmacy, replica goods, cheap software, loans, stock promotion, adult content, SEO and marketing offers.

LEGITIMATE means genuine communication: personal or work correspondence, discussions on mailing lists, project and technical messages, real notifications and transactional emails, and newsletters the recipient plausibly subscribed to.

Notes:
- Only the text of the email is available. It may start with the subject line. Headers and attachments are not available.
- Values such as links and addresses were replaced with placeholders: [URL], [EMAIL_ADDRESS], [DOMAIN], [PHONE_NUMBER], [YEAR], [FILE_PATH]. A placeholder alone is not evidence either way.
- Judge the intent and content of the message, not its formatting, length or language quality alone. A polite, well written email can still be fraud.

Answer with exactly one word: UNWANTED or LEGITIMATE."""

ANSWERS = {1: "UNWANTED", 0: "LEGITIMATE"}


def email_message(text):
    return f"Email:\n<<<\n{text[:MAX_EMAIL_CHARS]}\n>>>"


def build_messages(system_prompt, text, examples=()):
    messages = [{"role": "system", "content": system_prompt}]
    for example_text, label in examples:
        messages += [{"role": "user", "content": email_message(example_text)},
                     {"role": "assistant", "content": ANSWERS[int(label)]}]
    return messages + [{"role": "user", "content": email_message(text)}]


def _answer_score(logprobs):
    # gpt-oss zwraca też tokeny rozumowania i znaczniki kanałów. Odpowiedź zaczyna się po ostatnim <|message|>.
    markers = [index for index, position in enumerate(logprobs) if position["token"] == "<|message|>"]
    if markers:
        logprobs = logprobs[markers[-1] + 1:]
    for position in logprobs:
        if not position["token"].strip():
            continue
        candidates = position.get("top_logprobs") or [position]
        mass = {"u": 0.0, "l": 0.0}
        for candidate in candidates:
            first = candidate["token"].strip().lower()[:1]
            if first in mass:
                mass[first] += math.exp(candidate["logprob"])
        if mass["u"] == mass["l"] == 0:
            return None
        floor = min(candidate["logprob"] for candidate in candidates)
        log_u = math.log(mass["u"]) if mass["u"] > 0 else floor
        log_l = math.log(mass["l"]) if mass["l"] > 0 else floor
        return 1 / (1 + math.exp(log_l - log_u))
    return None


def classify(model, messages, think=None, top_logprobs=20, timeout=600):
    payload = {
        "model": model, "messages": messages, "stream": False, "logprobs": True, "top_logprobs": top_logprobs,
        "options": {"temperature": 0, "seed": 42, "num_ctx": NUM_CTX, "num_predict": 4 if think in (None, False) else 1024},
        "keep_alive": "30m",
    }
    if think is not None:
        payload["think"] = think
    start = time.perf_counter()
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    if response.status_code == 400 and "think" in payload and "think" in response.text.lower():
        del payload["think"]  # model bez trybu rozumowania
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    result = response.json()
    answer = result["message"]["content"].strip()
    score = _answer_score(result.get("logprobs") or [])
    if score is None:  # brak rozkładu tokenów: wynik z treści odpowiedzi
        word = answer.lower()
        score = 1.0 if word.startswith("unwanted") else 0.0 if word.startswith("legitimate") else 0.5
    return {
        "answer": answer[:40], "score": score, "valid": answer.upper().startswith(("UNWANTED", "LEGITIMATE")),
        "seconds": time.perf_counter() - start, "prompt_tokens": result.get("prompt_eval_count"),
        "thinking_chars": len(result["message"].get("thinking") or ""),
    }

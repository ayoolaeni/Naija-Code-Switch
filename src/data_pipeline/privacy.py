"""
Privacy scrubbing for retrieved text (README §10): strip emails, phone
numbers, social handles, and URLs before anything is stored in
data/processed/.
"""
import re

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HANDLE_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{2,}")
# Nigerian mobile numbers (+234/0 prefixed, 10-11 digits) and generic long
# digit runs that look like phone numbers.
_PHONE_RE = re.compile(r"(?:\+?234|0)[\d\-\s]{9,13}\d")


def scrub(text: str) -> str:
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _URL_RE.sub("[URL]", text)
    text = _HANDLE_RE.sub("[HANDLE]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    return text

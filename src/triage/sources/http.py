import time

RETRIES = 3


class FetchError(RuntimeError):
    pass


def get_json(session, url: str, params: dict | None = None, delay: float = 1.0) -> dict:
    for attempt in range(RETRIES):
        resp = session.get(url, params=params)
        if resp.status_code == 200:
            return resp.json()
        if attempt < RETRIES - 1:
            time.sleep(delay * (2 ** attempt))
    raise FetchError(f"{url} failed after {RETRIES} attempts (last status {resp.status_code})")

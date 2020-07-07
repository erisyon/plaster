import json
import http.client
from retrying import retry as _retry
from urllib.parse import urlsplit
from plaster.tools.log.log import error


class HTTPNonSuccessStatus(ValueError):
    def __init__(self, code, url):
        self.code = code
        self.url = url


def http_method(url, method="GET", body="", headers={}, n_retries=0, **kwargs):
    """
    Simple url caller, avoids request library.

    Rules:
        Raises HTTPNonSuccessStatus on anything but 2XX
        Retries (with reasonable backoff) up to retry
        Passes kwargs to the HTTP Connection Class
        Uses Content-Length if provided
        Encodes to UTF-8 if not application/octet-stream
        Returns a dict from json.loads if application/json
        Returns str in any other cases
    """

    urlp = urlsplit(url)

    if urlp.scheme == "http":
        conn = http.client.HTTPConnection(urlp.netloc, **kwargs)
    elif urlp.scheme == "https":
        conn = http.client.HTTPSConnection(urlp.netloc, **kwargs)
    else:
        raise TypeError("Unknown protocol")

    def without_retry():
        conn.request(method, urlp.path + "?" + urlp.query, body=body, headers=headers)
        response = conn.getresponse()
        if str(response.status)[0] != "2":
            raise HTTPNonSuccessStatus(response.status, url)
        return response

    @_retry(
        retry_on_exception=lambda e: isinstance(e, HTTPNonSuccessStatus)
        and str(e.code)[0] != "3",
        wait_exponential_multiplier=100,
        wait_exponential_max=500,
        stop_max_attempt_number=n_retries,
    )
    def with_retry():
        return without_retry()

    try:
        if n_retries > 0:
            response = with_retry()
        else:
            response = without_retry()
    except Exception as e:
        error(
            f"\nFailure during http request:\n"
            f"  domain={urlp.scheme}://{urlp.netloc}\n"
            f"  method={method}\n"
            f"  urlp.path={urlp.path}\n"
            f"  urlp.query={urlp.query}\n"
            f"  body={body}\n"
            f"  headers={headers}\n"
        )
        raise e

    if response.getheader("Content-Length") is not None:
        length = int(response.getheader("Content-Length"))
        result = response.read(length)
    else:
        result = response.read()

    if response.getheader("Content-Type") != "application/octet-stream":
        result = result.decode("utf-8")

    if response.getheader("Content-Type") == "application/json":
        result = json.loads(result)

    return result

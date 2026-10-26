# Project Titan Client Implementation Guide

## Retry Behavior
The retrieval client retries transient request failures with exponential backoff. It does not retry authorization failures or invalid query payloads. The retry count is bounded to avoid amplifying an outage, and each attempt keeps the same request ID for tracing.

```python
import random
import time


def retry_retrieval(request, max_attempts=4, base_delay=0.2):
    """Retry a transient retrieval request with bounded exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return request()
        except TimeoutError:
            if attempt + 1 >= max_attempts:
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay + random.uniform(0, delay / 10))


def is_transient_status(status_code):
    """Return whether a response status is safe to retry."""
    return status_code in {408, 429, 500, 502, 503, 504}
```

## Request Metadata
Every retrieval request carries a request identifier, tenant identifier, allowed collection names, and a time budget. The tenant scope is applied before context construction. Model prompts must never grant access to data that was excluded by application authorization.

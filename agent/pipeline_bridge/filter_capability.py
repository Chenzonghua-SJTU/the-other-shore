"""Bounded loopback-only capability checks, without DNS/proxies/redirects."""
import http.client
import json
import time
from urllib.parse import urlsplit

from .trajectory_filter import capability


def advertise(pool):
    health = pool.health
    def enhanced_health():
        result = health()
        receipts = []
        with pool.lock:
            for worker in pool.workers:
                if worker is not None and worker.ready():
                    try:
                        path = worker.directory/'ready.json'
                        if path.stat().st_size <= 16384:
                            receipts.append(json.loads(path.read_text()).get('trajectory_filter'))
                    except (OSError, ValueError):
                        pass
        expected = capability()
        result['trajectory_filter_status'] = {
            'expected': expected, 'authenticated_workers': sum(r == expected for r in receipts),
            'pending_workers': len(pool.workers)-len(receipts),
            'mismatched_workers': sum(r != expected for r in receipts)}
        if len(receipts) == len(pool.workers) and all(r == expected for r in receipts):
            result['trajectory_filter'] = capability()
        return result
    pool.health = enhanced_health


def verify(endpoint, *, timeout=3., wait_timeout=0.):
    parsed = urlsplit(endpoint)
    if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1')
            or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment
            or parsed.path not in ('', '/', '/search') or parsed.port is None):
        raise ValueError('Invalid filter worker endpoint')
    started = time.monotonic()
    deadline = started+max(0., wait_timeout)
    while True:
        connection = http.client.HTTPConnection('::1' if parsed.hostname == '::1' else '127.0.0.1', parsed.port,
                                                timeout=min(timeout, max(.01, deadline-time.monotonic())) if wait_timeout else timeout)
        try:
            connection.request('GET', '/health')
            response = connection.getresponse()
            content = response.read(65537)
            if response.status != 200 or len(content) > 65536:
                raise ValueError('Invalid worker health response')
            health = json.loads(content)
        finally:
            connection.close()
        if health.get('trajectory_filter') == capability():
            return {'endpoint': endpoint, 'verified_capability': capability(), 'pool_size': health.get('pool_size'),
                    'capability_wait_seconds': time.monotonic()-started}
        status = health.get('trajectory_filter_status', {})
        pending = (status.get('expected') == capability() and status.get('pending_workers', 0) > 0
                   and status.get('mismatched_workers') == 0)
        if pending and time.monotonic() < deadline:
            time.sleep(min(.1, max(0., deadline-time.monotonic())))
            continue
        if pending:
            raise ValueError('Filter consumer warmup/restart capability wait exhausted')
        raise ValueError('Worker filter consumer version/source hash mismatch or capability missing')

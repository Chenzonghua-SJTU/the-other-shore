"""Loopback-only research console. Run: python -m shore_agent.web --port 8765.

The browser gets a bounded application API, never filesystem or simulator access.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from shore_worlds import public_worlds, public_world_sets
from shore_rules import rules_for

PROJECT = Path(__file__).resolve().parents[1]
MAX_BODY = 64 * 1024
RUN_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$')
STATIC = {'/': ('index.html', 'text/html; charset=utf-8'),
          '/baselines': ('baselines.html', 'text/html; charset=utf-8'),
          '/baselines.js': ('baselines.js', 'text/javascript; charset=utf-8'),
          '/baselines.css': ('baselines.css', 'text/css; charset=utf-8'),
          '/index.html': ('index.html', 'text/html; charset=utf-8'),
          '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
          '/styles.css': ('styles.css', 'text/css; charset=utf-8')}


class WebServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def server_close(self):
        try:
            manager = getattr(self, 'manager', None)
            if manager is not None:
                manager.close()
        finally:
            super().server_close()


class Handler(BaseHTTPRequestHandler):
    server_version = 'ShoreLocal/1.0'

    def log_message(self, *_args):
        # Never persist request bodies, credentials, arbitrary URLs or model text.
        pass

    def _reply(self, status, payload, content_type='application/json; charset=utf-8', extra=None):
        if not isinstance(payload, bytes):
            payload = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Content-Security-Policy',
                         "default-src 'self'; script-src 'self'; style-src 'self'; "
                         "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _error(self, status, code):
        self._reply(status, {'error': code})

    def _trusted_request(self):
        port = self.server.server_port
        allowed = {f'127.0.0.1:{port}', f'localhost:{port}'}
        host = self.headers.get('Host', '').lower()
        if host not in allowed:
            self._error(403, 'invalid_host')
            return False
        if self.headers.get('Sec-Fetch-Site', '') == 'cross-site':
            self._error(403, 'cross_site_request')
            return False
        origin = self.headers.get('Origin')
        if origin is not None and origin.lower() != 'http://' + host:
            self._error(403, 'invalid_origin')
            return False
        return True

    def _route(self):
        try:
            parsed = urlsplit(self.path)
        except ValueError:
            return None
        # Encoded separators/traversal never become filesystem paths.
        if parsed.scheme or parsed.netloc or '%' in parsed.path or '..' in parsed.path:
            return None
        return parsed.path

    def do_GET(self):
        if not self._trusted_request():
            return
        path = self._route()
        try:
            if path in STATIC:
                filename, mime = STATIC[path]
                file = PROJECT / 'web' / filename
                if not file.is_file():
                    return self._error(404, 'page_not_found')
                return self._reply(200, file.read_bytes(), mime)
            if path == '/api/bootstrap':
                listing = self.server.manager.list_runs()
                return self._reply(200, dict(listing, token=self.server.session_token,
                    worlds=public_worlds(), world_sets=public_world_sets(),
                    **self.server.manager.client_configuration()))
            if path == '/api/runs':
                return self._reply(200, self.server.manager.list_runs())
            if path == '/api/baselines':
                batches = []
                experiment_root = PROJECT / 'experiments'
                sources = sorted((experiment_root / 'E3').rglob('summary.json'))
                sources += sorted(experiment_root.glob('v15-local-*/summary.json'))
                sources += sorted((experiment_root / 'E4').rglob('summary.json'))
                sources += sorted((experiment_root / 'E5').rglob('summary.json'))
                seen = set()
                for source in sources:
                    try:
                        data = json.loads(source.read_text())
                        if data['experiment_id'] in seen:
                            continue
                        model = data['configuration']['model']
                        environment = data['configuration'].get('environment', {})
                        profile = rules_for(environment)
                        ruleset = profile['ruleset']
                        group = data.get('experiment_group', ruleset.upper())
                        seen.add(data['experiment_id'])
                        batches.append(dict(id=data['experiment_id'], status=data['status'],
                            experiment_group=group, ruleset=ruleset,
                            protocol=data.get('protocol', {}),
                            energy_budget=data.get('protocol', {}).get('energy_budget', profile['energy_budget']),
                            wall_seconds=environment.get('wall_seconds', profile['wall_seconds']),
                            noise=environment.get('noise', False), output_limit=model.get('max_tokens_per_call'),
                            diagnostics_enabled=data['configuration'].get('numerics', {}).get('diagnostics_enabled', False),
                            diagnostics_harmonic=data['configuration'].get('numerics', {}).get('diagnostics_harmonic', False),
                            diagnostics_on_fit=data['configuration'].get('numerics', {}).get('diagnostics_on_fit', False),
                            symbolic_regression_enabled=data['configuration'].get('numerics', {}).get('symbolic_regression_enabled', False),
                            guidance_profile=data['configuration'].get('agent', {}).get('guidance_profile', 'baseline'),
                            loop_recovery=data['configuration'].get('agent', {}).get('loop_recovery', False),
                            max_total_tokens=model.get('max_total_tokens'),
                            model_sampling_seed=model.get('sampling_seed'),
                            model_temperature=model.get('temperature'),
                            updated_at=data['updated_at'], model=model['model'],
                            thinking=model['thinking'], deployment=model['deployment'],
                            worlds=[{key: row.get(key) for key in (
                                'world_id', 'run_id', 'status', 'reason', 'calls', 'tokens',
                                'completion_tokens', 'conservative_reserved_tokens',
                                'wall_seconds_including_evaluation',
                                'exploration_seconds', 'evaluation_seconds', 'exploration_deadline_met',
                                'average_output_tokens_per_call', 'max_call_output_tokens',
                                'output_limit_exceeded_calls', 'output_cap_compliant',
                                'requests_above_4096',
                                'requests_without_reported_usage',
                                'symbolic_search_calls', 'symbolic_search_seconds', 'symbolic_search_completed_calls',
                                'energy_used', 'evaluation_complete', 'evaluation_score',
                                'independent_rmse', 'n_test_trajectories')}
                                for row in data['worlds']]))
                    except (OSError, ValueError, KeyError):
                        continue
                return self._reply(200, {'batches': batches})
            pieces = (path or '').strip('/').split('/')
            if (len(pieces) == 7 and pieces[:2] == ['api', 'runs'] and RUN_ID.fullmatch(pieces[2])
                    and pieces[3] == 'code' and pieces[5] == 'artifacts'):
                content, mime = self.server.manager.code_artifact(pieces[2], pieces[4], pieces[6])
                return self._reply(200, content, mime, extra={
                    'Content-Disposition': f'attachment; filename="{pieces[6]}"'})
            if len(pieces) in (3, 4) and pieces[:2] == ['api', 'runs'] and RUN_ID.fullmatch(pieces[2]):
                if len(pieces) == 4 and pieces[3] != 'export':
                    return self._error(404, 'not_found')
                snapshot = self.server.manager.snapshot(pieces[2])
                if len(pieces) == 4:
                    return self._reply(200, snapshot, extra={
                        'Content-Disposition': f'attachment; filename="{pieces[2]}.json"'})
                return self._reply(200, snapshot)
            self._error(404, 'not_found')
        except (FileNotFoundError, KeyError):
            self._error(404, 'run_not_found')
        except ValueError:
            self._error(400, 'invalid_request')
        except Exception:
            self._error(500, 'server_error')

    def do_POST(self):
        self.close_connection = True
        if not self._trusted_request():
            return
        token = self.headers.get('X-Session-Token', '')
        if not token.isascii() or not secrets.compare_digest(token, self.server.session_token):
            return self._error(403, 'invalid_session_token')
        if self.headers.get('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
            return self._error(415, 'json_required')
        try:
            lengths = self.headers.get_all('Content-Length') or []
            if len(lengths) != 1 or self.headers.get('Transfer-Encoding'):
                return self._error(400, 'invalid_content_length')
            try:
                length = int(lengths[0])
            except ValueError:
                return self._error(400, 'invalid_content_length')
            if length < 0:
                return self._error(400, 'invalid_content_length')
            if length > MAX_BODY:
                return self._error(413, 'request_too_large')
            self.connection.settimeout(10)
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                return self._error(400, 'json_object_required')
            json.dumps(body, allow_nan=False)
        except (ValueError, UnicodeError, OSError, RecursionError):
            return self._error(400, 'invalid_json')
        path = self._route()
        try:
            if path == '/api/runs':
                return self._reply(202, {'run_id': self.server.manager.start(body)})
            parts = (path or '').strip('/').split('/')
            if (len(parts) == 4 and parts[:2] == ['api', 'runs'] and parts[-1] in {'stop', 'delete', 'restore'}
                    and RUN_ID.fullmatch(parts[2])):
                action = parts[-1]
                if body:
                    return self._error(400, action + '_takes_no_arguments')
                result = getattr(self.server.manager, action)(parts[2])
                return self._reply(200, {'ok': True} if action == 'stop' else result)
            return self._error(404, 'not_found')
        except (FileNotFoundError, KeyError):
            self._error(404, 'run_not_found')
        except ValueError as exc:
            # Only locally generated validation codes; never echo arbitrary exception text.
            code = getattr(exc, 'code', None)
            self._error(400, code if isinstance(code, str) and re.fullmatch('[a-z_]{1,80}', code) else 'invalid_run_configuration')
        except RuntimeError as exc:
            code = getattr(exc, 'code', None)
            self._error(409, code if isinstance(code, str) and re.fullmatch('[a-z_]{1,80}', code) else 'run_already_active')
        except Exception:
            self._error(500, 'server_error')
        finally:
            body.pop('api_key', None)

    def do_OPTIONS(self):
        self._error(403, 'cross_origin_access_disabled')


def make_server(host='127.0.0.1', port=0, manager=None, config_path=None):
    if host not in ('127.0.0.1', 'localhost'):
        raise ValueError('The research console must bind to loopback')
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError('Port must be an integer from 0 through 65535')
    if manager is None:
        from .web_jobs import RunManager
        manager = RunManager(PROJECT, config_path=config_path)
        manager.client_configuration()  # Fail before listening if startup config is invalid.
    server = WebServer((host, port), Handler)
    server.manager = manager
    server.session_token = secrets.token_urlsafe(32)
    return server


def main():
    parser = argparse.ArgumentParser(description='Local physical-law discovery console')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--config', type=Path, help='Trusted server JSON settings; overrides config.example.json')
    parser.add_argument('--prompt-api-key', action='store_true', help='Read the model key without echo; keep it only in this process')
    args = parser.parse_args()
    if args.config is None and (PROJECT / 'config.local.web.json').is_file():
        args.config = PROJECT / 'config.local.web.json'
    server = make_server(port=args.port, config_path=args.config)
    if args.prompt_api_key:
        import getpass
        from .model import model_auth_config
        key_env, _ = model_auth_config(server.manager._server_config()['model'])
        key = getpass.getpass('Model API key (hidden): ').strip()
        if not key or len(key) > 512 or any(ord(char) < 32 or ord(char) == 127 for char in key):
            server.server_close()
            parser.error('Invalid API key format')
        os.environ[key_env] = key
        del key
    print(f'彼岸研究台已启动：http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

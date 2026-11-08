"""Preserve a confirmed Console submission while owned services shut down."""
import json
import signal
from pathlib import Path


class SubmissionLifecycle:
    def __init__(self):
        self.confirmed = False
        self.post_submit_sigterm_count = 0

    def observe(self, environment):
        original_call = environment.call

        def call(method, **kwargs):
            reply = original_call(method, **kwargs)
            # Only the authenticated Console reply can acknowledge delivery.
            # A model-selected draft, local status or an existing file cannot.
            if (method == 'submit' and isinstance(reply, dict)
                    and reply.get('accepted') is True
                    and reply.get('submitted') is True):
                self.confirmed = True
            return reply

        environment.call = call
        return environment

    def handle_sigterm(self, signum, frame):
        if signum == signal.SIGTERM and self.confirmed:
            self.post_submit_sigterm_count += 1
            # The runner sends TERM two seconds after accepting the submission.
            # Let the existing context managers finish killing/reaping their
            # owned Julia/vLLM groups, then main returns its normal success code.
            return
        raise SystemExit(128 + signum)

    def write_receipt(self, path):
        try:
            Path(path).write_text(json.dumps({
                'confirmed_console_submission': self.confirmed,
                'post_submit_sigterm_count': self.post_submit_sigterm_count,
                'confirmation_source': 'Console submit accepted=true and submitted=true',
                'post_submit_signal_policy': 'finish_owned_cleanup_then_normal_exit',
            }, indent=2) + '\n')
        except OSError:
            # Optional diagnostics must not change the actual exit outcome.
            pass

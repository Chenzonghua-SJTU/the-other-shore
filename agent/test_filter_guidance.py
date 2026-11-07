"""Prompt-only variant: no model, Julia, or numerical filter execution."""
import copy
import json
import tempfile
from pathlib import Path
import unittest

from .pipeline_bridge.filter_guidance import configure, MARKER, OLD_DESCRIPTION, NEW_DESCRIPTION, GUIDANCE
from .pipeline_bridge.reliability import ProtocolClient, OVERRIDE


class LoggedClient:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.attribute = 'forwarded'

    def complete(self, messages, **kwargs):
        with (self.directory/'model_messages.jsonl').open('a') as stream:
            stream.write(json.dumps({'messages': messages, 'kwargs': kwargs}, ensure_ascii=False)+'\n')
        return {'result': 'unchanged'}


class FilterGuidanceTests(unittest.TestCase):
    def test_disabled_identity_and_exact_messages(self):
        with tempfile.TemporaryDirectory() as root:
            client = LoggedClient(root)
            wrapped = configure(client, enabled=False)
            self.assertIs(wrapped, client)
            messages = [{'role': 'system', 'content': OLD_DESCRIPTION}, {'role': 'user', 'content': 'data'}]
            before = copy.deepcopy(messages)
            wrapped.complete(messages, tools=[{'name': 'unchanged'}])
            self.assertEqual(messages, before)
            logged = json.loads((Path(root)/'model_messages.jsonl').read_text())
            self.assertEqual(logged['messages'], before)

    def test_enabled_actual_request_logged_and_idempotent(self):
        with tempfile.TemporaryDirectory() as root:
            client = LoggedClient(root)
            wrapped = configure(ProtocolClient(client), enabled=True)
            self.assertEqual(wrapped.attribute, 'forwarded')
            messages = [{'role': 'system', 'content': 'base。'+OLD_DESCRIPTION}, {'role': 'user', 'content': 'raw observations'}]
            schemas = [{'type': 'function', 'function': {'name': 'existing_tool'}}]
            before = copy.deepcopy(schemas)
            self.assertEqual(wrapped.complete(messages, tools=schemas), {'result': 'unchanged'})
            self.assertNotIn(OLD_DESCRIPTION, messages[0]['content'])
            self.assertIn(NEW_DESCRIPTION, messages[0]['content'])
            self.assertIn(OVERRIDE, messages[0]['content'])
            self.assertIn(MARKER, messages[0]['content'])
            first = copy.deepcopy(messages)
            wrapped.complete(messages, tools=schemas)
            self.assertEqual(first, messages)
            self.assertEqual(messages[1], {'role': 'user', 'content': 'raw observations'})
            self.assertEqual(schemas, before)
            logged = [json.loads(line) for line in (Path(root)/'model_messages.jsonl').read_text().splitlines()]
            self.assertEqual(logged[-1]['messages'], messages)
            self.assertEqual(messages[0]['content'].count(MARKER), 1)

    def test_disabled_bf2_prompt_schemas_byte_equal(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            messages = [{'role': 'system', 'content': 'base。'+OLD_DESCRIPTION}]
            schemas = [{'name': 'symbolic_regression', 'parameters': {'train_probe_ids': ['p']}}]
            ProtocolClient(LoggedClient(a)).complete(copy.deepcopy(messages), tools=copy.deepcopy(schemas))
            configure(ProtocolClient(LoggedClient(b)), enabled=False).complete(copy.deepcopy(messages), tools=copy.deepcopy(schemas))
            self.assertEqual((Path(a)/'model_messages.jsonl').read_bytes(), (Path(b)/'model_messages.jsonl').read_bytes())

    def test_guidance_only_changes_system_and_is_optional(self):
        with tempfile.TemporaryDirectory() as root:
            client = configure(LoggedClient(root), enabled=True)
            messages = [{'role': 'user', 'content': OLD_DESCRIPTION}]
            before = copy.deepcopy(messages);client.complete(messages);self.assertEqual(messages, before)
            client.complete([])
            self.assertIn('12个观测点', GUIDANCE)
            self.assertIn('不保证', GUIDANCE)
            self.assertIn('省略preprocessing_id', GUIDANCE)

    def test_invalid_flag_rejected(self):
        for enabled in [None, 1, 0, 'true']:
            with self.assertRaises(ValueError):configure(None, enabled=enabled)


if __name__ == '__main__':unittest.main()

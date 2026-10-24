# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fortanix CCM/CDH vault-key retrieval in the measured guest."""

import base64
import contextlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from cvm.common.errors import BuildError
from cvm.key_release.contracts import FORTANIX_LUKS_TOKEN_TYPE
from cvm.key_release.fortanix import runtime


class FortanixRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.object_id = "00000000-0000-4000-8000-000000000002"
        self.config = {
            "attestation_agent": "/usr/lib/cvm/bin/attestation-agent",
            "cdh_oneshot": "/usr/lib/cvm/bin/cdh-oneshot",
            "fortanix_ccm": {
                "dsm_endpoint": "https://dsm.test",
                "dsm_app_id": "00000000-0000-4000-8000-000000000003",
                "ccm_appconfig_id": "a" * 32,
                "ccm_domain_names": ["ccm.test", "ccm-backup.test"],
            },
        }
        self.token = {
            "type": FORTANIX_LUKS_TOKEN_TYPE,
            "keyslots": ["0"],
            "schema_version": 1,
            "provider": "fortanix_ccm",
            "dsm_object_id": self.object_id,
        }

    def test_fresh_attestation_agent_and_cdh_release_the_bound_secret(self):
        secret = bytes(range(64))
        process = MagicMock()
        process.poll.return_value = None
        captured = {}

        def start(command, **kwargs):
            socket = Path(command[2].removeprefix("unix://"))
            socket.touch()
            captured["aa_command"] = command
            captured["aa_config"] = Path(command[4]).read_text()
            captured["aa_environment"] = kwargs["env"]
            return process

        def execute(command, **kwargs):
            captured["cdh_command"] = command
            captured["cdh_config"] = Path(command[2]).read_text()
            captured["cdh_environment"] = kwargs["env"]
            return base64.b64encode(secret)

        hostile_environment = {
            "AA_KBC_PARAMS": "other::https://attacker.test",
            "AA_SOCKET": "unix:///tmp/attacker.sock",
            "CCM_KBC_APP_ID": "00000000-0000-4000-8000-000000000099",
            "CCM_KBC_DSM_ENDPOINT": "https://attacker.test",
        }
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.dict(runtime.os.environ, hostile_environment),
                patch.object(runtime, "RUNTIME", Path(directory)),
                patch.object(runtime.subprocess, "Popen", side_effect=start),
                patch.object(runtime, "run", side_effect=execute),
                patch.object(runtime, "memory_file", side_effect=lambda value: contextlib.nullcontext(29)),
            ):
                with runtime.authorized_key(self.config, bytes(32), self.token, budget=30) as key_fd:
                    self.assertEqual(key_fd, 29)

        self.assertEqual(
            captured["cdh_command"][-3:],
            ["get-resource", "--resource-uri", "kbs:///dsm/key/" + self.object_id],
        )
        self.assertIn('name = "ccm_kbc"', captured["cdh_config"])
        self.assertIn('url = "https://dsm.test"', captured["cdh_config"])
        self.assertIn('dsm_app_id = "00000000-0000-4000-8000-000000000003"', captured["cdh_config"])
        self.assertIn('ccm_appconfig_id = "' + "a" * 32 + '"', captured["aa_config"])
        self.assertIn('ccm_domain_names = ["ccm.test","ccm-backup.test"]', captured["aa_config"])
        self.assertTrue(runtime.FORTANIX_CONTROL_ENVIRONMENT.isdisjoint(captured["aa_environment"]))
        self.assertTrue(runtime.FORTANIX_CONTROL_ENVIRONMENT.isdisjoint(captured["cdh_environment"]))
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=5)

    def test_unbound_or_modified_token_never_starts_attestation(self):
        for token in (None, dict(self.token, dsm_object_id="other"), dict(self.token, extra=True)):
            with (
                self.subTest(token=token),
                patch.object(runtime.subprocess, "Popen") as start,
                self.assertRaises(BuildError),
            ):
                with runtime.authorized_key(self.config, bytes(32), token, budget=30):
                    pass
            start.assert_not_called()


if __name__ == "__main__":
    unittest.main()

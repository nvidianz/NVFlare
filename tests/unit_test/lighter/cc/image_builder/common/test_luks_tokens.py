# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Authenticated key-release metadata in the frozen LUKS2 header."""

import contextlib
import json
import unittest
from unittest.mock import patch

from cvm.common.errors import BuildError
from cvm.common.luks import install_key_release_token, key_release_token
from cvm.key_release.contracts import FORTANIX_LUKS_TOKEN_TYPE


class LuksTokenTests(unittest.TestCase):
    def setUp(self):
        self.token = {
            "type": FORTANIX_LUKS_TOKEN_TYPE,
            "keyslots": ["0"],
            "schema_version": 1,
            "provider": "fortanix_ccm",
            "dsm_object_id": "00000000-0000-4000-8000-000000000002",
        }

    def test_only_the_canonical_fortanix_token_is_accepted(self):
        self.assertEqual(key_release_token({"tokens": {"0": self.token}}), self.token)
        for tokens in (
            {"1": self.token},
            {"0": dict(self.token, provider="trustee")},
            {"0": dict(self.token, dsm_object_id="not-a-uuid")},
            {"0": dict(self.token, extra=True)},
        ):
            with self.subTest(tokens=tokens), self.assertRaises(BuildError):
                key_release_token({"tokens": tokens})

    def test_token_is_imported_before_the_header_is_frozen(self):
        captured = {}

        @contextlib.contextmanager
        def private_file(data):
            captured["data"] = data
            yield 31

        with (
            patch("cvm.common.luks.memory_file", side_effect=private_file),
            patch("cvm.common.luks.run") as execute,
            patch("cvm.common.luks.inspect_header", return_value={"tokens": {"0": self.token}}),
        ):
            install_key_release_token("/dev/vault", self.token)

        self.assertEqual(json.loads(captured["data"]), self.token)
        self.assertEqual(
            execute.call_args.args[0],
            [
                "cryptsetup",
                "token",
                "import",
                "--token-id",
                "0",
                "--json-file",
                "/proc/self/fd/31",
                "/dev/vault",
            ],
        )
        self.assertEqual(execute.call_args.kwargs, {"pass_fds": (31,), "secret": True})


if __name__ == "__main__":
    unittest.main()

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Provenance for the pinned Fortanix CoCo guest components."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cvm.common.errors import BuildError
from cvm.common.io import digest_file
from cvm.common.versions import FORTANIX_GUEST_COMPONENTS_COMMIT
from cvm.key_release.fortanix.provenance import provenance


class FortanixProvenanceTests(unittest.TestCase):
    def test_record_requires_clean_source_and_both_provider_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aa = root / "ttrpc-aa"
            cdh = root / "cdh-oneshot"
            aa.write_bytes(b"aa")
            cdh.write_bytes(b"cdh")
            responses = [
                FORTANIX_GUEST_COMPONENTS_COMMIT + "\n",
                "",
                "token plugins: CCM_AS\n",
                "ccm_kbc: enabled\n",
            ]
            with patch(
                "cvm.key_release.fortanix.provenance.subprocess.check_output", side_effect=responses
            ):
                record = provenance(root, aa, cdh)
            self.assertEqual(record["source_commit"], FORTANIX_GUEST_COMPONENTS_COMMIT)
            self.assertEqual(record["attestation_agent_sha256"], digest_file(aa))
            self.assertEqual(record["cdh_oneshot_sha256"], digest_file(cdh))

            for responses, diagnostic in (
                (["f" * 40 + "\n"], "reviewed"),
                ([FORTANIX_GUEST_COMPONENTS_COMMIT + "\n", " M source\n"], "clean"),
                ([FORTANIX_GUEST_COMPONENTS_COMMIT + "\n", "", "no plugin\n"], "ccm_as"),
                (
                    [FORTANIX_GUEST_COMPONENTS_COMMIT + "\n", "", "CCM_AS\n", "ccm_kbc: disabled\n"],
                    "ccm_kbc",
                ),
            ):
                with (
                    self.subTest(diagnostic=diagnostic),
                    patch(
                        "cvm.key_release.fortanix.provenance.subprocess.check_output", side_effect=responses
                    ),
                    self.assertRaisesRegex(BuildError, diagnostic),
                ):
                    provenance(root, aa, cdh)


if __name__ == "__main__":
    unittest.main()

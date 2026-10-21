# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Key-release provider selection and fail-closed lifecycle tests."""

import unittest
from unittest.mock import patch

from cvm.common.errors import BuildError
from cvm.key_release.base import provider


class KeyReleaseProviderTests(unittest.TestCase):
    def test_trustee_adapter_delegates_without_changing_resource_identity(self):
        settings = {"url": "https://trustee.test", "ca": "ca.pem", "admin_token_file": "token.jwt"}
        release = provider({"provider": "trustee", "trustee": settings})
        secret = bytes(64)
        with (
            patch("cvm.key_release.trustee.admin.client.admin_token", return_value="token") as preflight,
            patch("cvm.key_release.trustee.admin.client.upload_resource") as upload,
            patch("cvm.key_release.trustee.admin.client.delete_resource") as delete,
        ):
            release.preflight()
            receipt = release.create_vault_binding(
                "keys/bundle/" + "a" * 64,
                secret,
                context={"vault_binding": "a" * 64},
            )
            release.verify_readback(receipt)
            release.revoke_vault_binding(receipt)
        preflight.assert_called_once_with(settings)
        upload.assert_called_once_with(settings, "keys/bundle/" + "a" * 64, secret)
        delete.assert_called_once_with(settings, "keys/bundle/" + "a" * 64)
        self.assertEqual(receipt["provider"], "trustee")

    def test_fortanix_fails_before_any_secret_can_be_created(self):
        release = provider({"provider": "fortanix_ccm", "fortanix_ccm": {}})
        with self.assertRaisesRegex(BuildError, "vault provisioning is unavailable"):
            release.preflight()

    def test_unknown_provider_has_no_fallback(self):
        with self.assertRaisesRegex(BuildError, "Unknown key release provider"):
            provider({"provider": "other", "other": {}})


if __name__ == "__main__":
    unittest.main()

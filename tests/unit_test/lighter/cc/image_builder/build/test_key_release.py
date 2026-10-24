# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Key-release provider selection and fail-closed lifecycle tests."""

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from cvm.common.errors import BuildError
from cvm.key_release.base import provider
from cvm.key_release.contracts import FORTANIX_LUKS_TOKEN_TYPE
from cvm.key_release.fortanix.admin import _DsmClient, _NoRedirect


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

    def test_fortanix_imports_and_binds_one_dsm_secret(self):
        group_id = "00000000-0000-4000-8000-000000000001"
        object_id = "00000000-0000-4000-8000-000000000002"
        release = provider(
            {
                "provider": "fortanix_ccm",
                "fortanix_ccm": {
                    "dsm": {
                        "endpoint": "https://dsm.test",
                        "group_id": group_id,
                        "provisioning_api_key_file": "/unused/dsm.key",
                    }
                },
            }
        )
        secret = bytes(range(64))
        context = {
            "deployment_id": "deployment",
            "build_id": "bundle",
            "platform": "intel_tdx",
        }
        with (
            patch.object(release.client, "preflight") as preflight,
            patch.object(release.client, "import_secret", return_value=object_id) as imported,
            patch.object(release.client, "update_metadata") as updated,
            patch.object(release.client, "inspect") as inspected,
            patch.object(release.client, "delete") as deleted,
        ):
            release.preflight()
            prepared = release.prepare_vault_binding(secret, context=context)
            token = release.runtime_token(prepared)
            receipt = release.create_vault_binding(
                "keys/bundle/" + "a" * 64,
                secret,
                context={"vault_binding": "a" * 64},
                prepared=prepared,
            )
            release.verify_readback(receipt)
            release.revoke_vault_binding(receipt)
            release.revoke_vault_binding(prepared)

        preflight.assert_called_once_with()
        self.assertEqual(imported.call_args.args[1], secret)
        self.assertNotIn(secret.hex(), repr(prepared))
        self.assertEqual(
            token,
            {
                "type": FORTANIX_LUKS_TOKEN_TYPE,
                "keyslots": ["0"],
                "schema_version": 1,
                "provider": "fortanix_ccm",
                "dsm_object_id": object_id,
            },
        )
        self.assertEqual(receipt["resource"], "keys/bundle/" + "a" * 64)
        self.assertEqual(receipt["state"], "active")
        updated.assert_called_once_with(object_id, receipt["metadata"])
        inspected.assert_called_once_with(object_id, receipt["metadata"])
        self.assertEqual(deleted.call_args_list, [call(object_id), call(object_id)])

    def test_unknown_provider_has_no_fallback(self):
        with self.assertRaisesRegex(BuildError, "Unknown key release provider"):
            provider({"provider": "other", "other": {}})

    def test_fortanix_dsm_redirects_are_never_followed(self):
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, "redirect", {}, "https://other.test"))

    def test_fortanix_dsm_import_uses_secret_object_contract(self):
        group_id = "00000000-0000-4000-8000-000000000001"
        object_id = "00000000-0000-4000-8000-000000000002"
        metadata = {"nvflare_state": "prepared"}
        secret = bytes(range(64))

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self, size):
                return json.dumps(
                    {
                        "kid": object_id,
                        "group_id": group_id,
                        "obj_type": "SECRET",
                        "enabled": True,
                        "key_ops": ["EXPORT", "APPMANAGEABLE"],
                        "custom_metadata": metadata,
                    }
                ).encode()

        with tempfile.TemporaryDirectory() as directory:
            credential = Path(directory) / "dsm.key"
            credential.write_text("YWJjZGVmZ2hpamtsbW5vcA==")
            client = _DsmClient(
                {
                    "endpoint": "https://dsm.test/",
                    "group_id": group_id,
                    "provisioning_api_key_file": str(credential),
                }
            )
            with patch("cvm.key_release.fortanix.admin.HTTP.open", return_value=Response()) as request:
                self.assertEqual(client.import_secret("vault", secret, metadata), object_id)

        submitted = request.call_args.args[0]
        self.assertEqual(submitted.method, "PUT")
        self.assertEqual(submitted.full_url, "https://dsm.test/crypto/v1/keys")
        self.assertEqual(submitted.get_header("Authorization"), "Basic YWJjZGVmZ2hpamtsbW5vcA==")
        body = json.loads(submitted.data)
        self.assertEqual(body["obj_type"], "SECRET")
        self.assertEqual(body["group_id"], group_id)
        self.assertEqual(body["key_ops"], ["EXPORT", "APPMANAGEABLE"])
        self.assertEqual(body["value"], base64.b64encode(secret).decode())

    def test_fortanix_dsm_readback_uses_sdk_descriptor_contract(self):
        group_id = "00000000-0000-4000-8000-000000000001"
        object_id = "00000000-0000-4000-8000-000000000002"
        metadata = {"nvflare_state": "active"}
        client = _DsmClient(
            {
                "endpoint": "https://dsm.test",
                "group_id": group_id,
                "provisioning_api_key_file": "/unused/dsm.key",
            }
        )
        response = {
            "kid": object_id,
            "group_id": group_id,
            "obj_type": "SECRET",
            "enabled": True,
            "key_ops": ["EXPORT", "APPMANAGEABLE"],
            "custom_metadata": metadata,
        }
        with patch.object(client, "request", return_value=response) as request:
            client.inspect(object_id, metadata)
        request.assert_called_once_with(
            "POST",
            "/crypto/v1/keys/info?view=json",
            {"Kid": object_id},
        )


if __name__ == "__main__":
    unittest.main()

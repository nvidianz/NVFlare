# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Adapter from the provider lifecycle to native CoCo Trustee APIs."""

from ...common.errors import require
from ...trustee import client


class TrusteeKeyReleaseProvider:
    name = "trustee"

    def __init__(self, settings):
        self.settings = settings

    def preflight(self):
        client.admin_token(self.settings)

    def create_vault_binding(self, resource, secret, *, context):
        client.upload_resource(self.settings, resource, secret)
        return {
            "schema_version": 1,
            "provider": self.name,
            "resource": resource,
            "state": "active",
        }

    def revoke_vault_binding(self, receipt):
        client.delete_resource(self.settings, receipt["resource"])

    def verify_readback(self, receipt):
        # Trustee v0.22 has no metadata/readback API that avoids releasing the
        # secret. A successful authenticated upload is its publication receipt.
        require(
            receipt.get("provider") == self.name and receipt.get("state") == "active",
            "Invalid Trustee vault receipt",
        )

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fortanix DSM lifecycle for one CVM vault secret.

CCM application/build approval and the DSM Trusted-CA runtime application are
deployment bootstrap operations. Stage 2 uses a least-privilege DSM API-key
application to import, inspect and delete vault secrets in that application's
configured group. The measured guest authenticates with a CCM-issued workload
certificate and retrieves the object through CoCo's ``ccm_kbc``.
"""

import base64
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from ...common.errors import BuildError, ConfigurationError, require
from ..contracts import FORTANIX_CCM, FORTANIX_LUKS_TOKEN_TYPE

MAX_RESPONSE_BYTES = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 30


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward a DSM API key to a redirect target."""

    def redirect_request(self, request, fp, code, message, headers, new_url):
        return None


HTTP = urllib.request.build_opener(_NoRedirect)


def _uuid(value, label):
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError, TypeError):
        raise BuildError(f"Invalid {label}") from None
    require(str(parsed) == value, f"Invalid {label}")
    return value


class _DsmClient:
    """Small DSM REST client with a deliberately narrow API surface."""

    def __init__(self, settings):
        self.endpoint = settings["endpoint"].rstrip("/")
        self.group_id = _uuid(settings["group_id"], "Fortanix DSM group id")
        self.credential = Path(settings["provisioning_api_key_file"])

    def _api_key(self):
        try:
            value = self.credential.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError):
            raise ConfigurationError("Cannot read the Fortanix DSM provisioning API key") from None
        # Fortanix API keys are already the base64 value used by HTTP Basic.
        if not (16 <= len(value) <= 4096 and re.fullmatch(r"[A-Za-z0-9_+/=-]+", value)):
            raise ConfigurationError("Invalid Fortanix DSM provisioning API key")
        return value

    def request(self, method, path, body=None):
        encoded = None if body is None else json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        request = urllib.request.Request(
            self.endpoint + path,
            data=encoded,
            method=method,
            headers={
                "Authorization": "Basic " + self._api_key(),
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        try:
            with HTTP.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                data = response.read(MAX_RESPONSE_BYTES + 1)
        except (OSError, urllib.error.HTTPError, urllib.error.URLError):
            raise BuildError(f"Fortanix DSM {method} request failed") from None
        require(len(data) <= MAX_RESPONSE_BYTES, "Fortanix DSM response is too large")
        if not data:
            return None
        try:
            value = json.loads(data)
        except (UnicodeError, json.JSONDecodeError):
            raise BuildError("Fortanix DSM returned an invalid response") from None
        require(isinstance(value, (dict, list)), "Fortanix DSM returned an invalid response")
        return value

    def preflight(self):
        query = urllib.parse.urlencode({"group_id": self.group_id, "limit": 1})
        require(isinstance(self.request("GET", "/crypto/v1/keys?" + query), list), "Fortanix DSM list failed")

    def import_secret(self, name, secret, metadata):
        value = self.request(
            "PUT",
            "/crypto/v1/keys",
            {
                "name": name,
                "description": "NVFlare CVM vault unlock secret",
                "obj_type": "SECRET",
                "value": base64.b64encode(secret).decode("ascii"),
                "group_id": self.group_id,
                "key_ops": ["EXPORT", "APPMANAGEABLE"],
                "custom_metadata": metadata,
                "enabled": True,
            },
        )
        require(isinstance(value, dict), "Fortanix DSM import returned an invalid response")
        object_id = _uuid(value.get("kid"), "Fortanix DSM object id")
        self._validate(value, object_id, metadata)
        return object_id

    def update_metadata(self, object_id, metadata):
        value = self.request(
            "PATCH",
            f"/crypto/v1/keys/{_uuid(object_id, 'Fortanix DSM object id')}",
            {"custom_metadata": metadata},
        )
        require(isinstance(value, dict), "Fortanix DSM update returned an invalid response")
        self._validate(value, object_id, metadata)

    def inspect(self, object_id, metadata):
        object_id = _uuid(object_id, "Fortanix DSM object id")
        # Match sdkms::OperationGetSobject: object lookup is a descriptor POST,
        # while PATCH and DELETE use the object-id path.
        value = self.request("POST", "/crypto/v1/keys/info?view=json", {"Kid": object_id})
        require(isinstance(value, dict), "Fortanix DSM readback returned an invalid response")
        self._validate(value, object_id, metadata)

    def delete(self, object_id):
        self.request("DELETE", f"/crypto/v1/keys/{_uuid(object_id, 'Fortanix DSM object id')}")

    def _validate(self, value, object_id, metadata):
        operations = value.get("key_ops")
        require(
            value.get("kid") == object_id
            and value.get("group_id") == self.group_id
            and value.get("obj_type") == "SECRET"
            and value.get("enabled") is True
            and isinstance(operations, list)
            and "EXPORT" in operations
            and value.get("custom_metadata") == metadata,
            "Fortanix DSM object does not match the requested vault binding",
        )


class FortanixCcmKeyReleaseProvider:
    name = FORTANIX_CCM

    def __init__(self, settings):
        self.settings = settings
        self.client = _DsmClient(settings["dsm"])

    def preflight(self):
        # Validate connectivity, credentials and group visibility before image
        # retrieval and before any vault secret exists.
        try:
            self.client.preflight()
        except BuildError:
            raise ConfigurationError("Fortanix DSM preflight failed") from None

    def prepare_vault_binding(self, secret, *, context):
        require(isinstance(secret, bytes) and len(secret) == 64, "Invalid vault secret")
        metadata = {
            "nvflare_schema": "1",
            "nvflare_state": "prepared",
            "deployment_id": context["deployment_id"],
            "cvm_build_id": context["build_id"],
            "platform": context["platform"],
        }
        suffix = uuid.uuid4().hex
        name = f"nvflare-{context['build_id']}-{context['platform']}-{suffix}"[:128]
        object_id = self.client.import_secret(name, secret, metadata)
        return {
            "schema_version": 1,
            "provider": self.name,
            "dsm_object_id": object_id,
            "dsm_group_id": self.client.group_id,
            "metadata": metadata,
            "state": "prepared",
        }

    def runtime_token(self, prepared):
        self._prepared(prepared)
        return {
            "type": FORTANIX_LUKS_TOKEN_TYPE,
            "keyslots": ["0"],
            "schema_version": 1,
            "provider": self.name,
            "dsm_object_id": prepared["dsm_object_id"],
        }

    def create_vault_binding(self, resource, secret, *, context, prepared=None):
        self._prepared(prepared)
        require(isinstance(secret, bytes) and len(secret) == 64, "Invalid vault secret")
        metadata = dict(
            prepared["metadata"],
            nvflare_state="active",
            resource=resource,
            vault_binding=context["vault_binding"],
        )
        self.client.update_metadata(prepared["dsm_object_id"], metadata)
        return dict(prepared, resource=resource, metadata=metadata, state="active")

    def revoke_vault_binding(self, receipt):
        self._recorded(receipt)
        self.client.delete(receipt["dsm_object_id"])

    def verify_readback(self, receipt):
        self._receipt(receipt)
        self.client.inspect(receipt["dsm_object_id"], receipt["metadata"])

    def _prepared(self, receipt):
        require(
            isinstance(receipt, dict)
            and receipt.get("schema_version") == 1
            and receipt.get("provider") == self.name
            and receipt.get("dsm_group_id") == self.client.group_id
            and receipt.get("state") == "prepared"
            and isinstance(receipt.get("metadata"), dict),
            "Invalid prepared Fortanix vault receipt",
        )
        _uuid(receipt.get("dsm_object_id"), "Fortanix DSM object id")

    def _receipt(self, receipt):
        require(
            isinstance(receipt, dict)
            and receipt.get("schema_version") == 1
            and receipt.get("provider") == self.name
            and receipt.get("dsm_group_id") == self.client.group_id
            and receipt.get("state") == "active"
            and isinstance(receipt.get("resource"), str)
            and isinstance(receipt.get("metadata"), dict),
            "Invalid Fortanix vault receipt",
        )
        _uuid(receipt.get("dsm_object_id"), "Fortanix DSM object id")

    def _recorded(self, receipt):
        require(
            isinstance(receipt, dict)
            and receipt.get("schema_version") == 1
            and receipt.get("provider") == self.name
            and receipt.get("dsm_group_id") == self.client.group_id
            and receipt.get("state") in ("prepared", "active"),
            "Invalid Fortanix vault receipt",
        )
        _uuid(receipt.get("dsm_object_id"), "Fortanix DSM object id")

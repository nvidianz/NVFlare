# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Backend-neutral lifecycle for attestation and vault-key release."""

from typing import Protocol

from ..common.errors import require
from .contracts import FORTANIX_CCM, PROVIDERS, TRUSTEE


class KeyReleaseProvider(Protocol):
    """Build-host operations needed to publish and revoke one vault secret."""

    name: str

    def preflight(self) -> None:
        """Validate credentials before retrieving an image or generating a key."""

    def create_vault_binding(self, resource: str, secret: bytes, *, context: dict) -> dict:
        """Publish a secret and return non-secret provider receipt fields."""

    def revoke_vault_binding(self, receipt: dict) -> None:
        """Remove access represented by a previously returned receipt."""

    def verify_readback(self, receipt: dict) -> None:
        """Verify that the provider state agrees with the receipt."""


def provider(settings: dict) -> KeyReleaseProvider:
    """Construct exactly the selected provider; there is no runtime fallback."""
    require(isinstance(settings, dict), "Invalid key release configuration")
    name = settings.get("provider")
    require(name in PROVIDERS, "Unknown key release provider")
    if name == TRUSTEE:
        from .trustee.admin import TrusteeKeyReleaseProvider

        return TrusteeKeyReleaseProvider(settings[TRUSTEE])
    from .fortanix.admin import FortanixCcmKeyReleaseProvider

    return FortanixCcmKeyReleaseProvider(settings[FORTANIX_CCM])

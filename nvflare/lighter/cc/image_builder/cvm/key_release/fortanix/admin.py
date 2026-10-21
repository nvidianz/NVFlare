# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fail-closed Fortanix CCM/DSM provider boundary.

The reviewed CoCo reference supplies ccm_as and ccm_kbc guest operations, but
does not publish a stable management API for creating a vault-bound AppConfig,
DSM Trusted-CA application, isolated group and object. Those operations cannot
be approximated without losing the vault-binding guarantee.
"""

from ...common.errors import BuildError, ConfigurationError

UNAVAILABLE = (
    "Fortanix CCM/DSM vault provisioning is unavailable: configure a released CoCo ccm_as/ccm_kbc revision "
    "and a supported CCM/DSM management API that binds one AppConfig and DSM authorization to this vault"
)


class FortanixCcmKeyReleaseProvider:
    name = "fortanix_ccm"

    def __init__(self, settings):
        self.settings = settings

    def preflight(self):
        # Fail before image retrieval and, critically, before generating a
        # plaintext vault secret. Never fall back to Trustee.
        raise ConfigurationError(UNAVAILABLE)

    def create_vault_binding(self, resource, secret, *, context):
        raise BuildError(UNAVAILABLE)

    def revoke_vault_binding(self, receipt):
        raise BuildError(UNAVAILABLE)

    def verify_readback(self, receipt):
        raise BuildError(UNAVAILABLE)

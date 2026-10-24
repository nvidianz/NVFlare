# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Provider identifiers shared by build-host and measured-guest code."""

from ..common.contracts import FORTANIX_LUKS_TOKEN_ID, FORTANIX_LUKS_TOKEN_TYPE

TRUSTEE = "trustee"
FORTANIX_CCM = "fortanix_ccm"
PROVIDERS = (TRUSTEE, FORTANIX_CCM)

__all__ = (
    "FORTANIX_CCM",
    "FORTANIX_LUKS_TOKEN_ID",
    "FORTANIX_LUKS_TOKEN_TYPE",
    "PROVIDERS",
    "TRUSTEE",
)

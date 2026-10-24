# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Record the exact reviewed Fortanix CoCo source and guest binaries."""

import subprocess

from ...common.errors import require
from ...common.io import digest_file
from ...common.versions import FORTANIX_GUEST_COMPONENTS_COMMIT

SOURCE_REPOSITORY = "https://github.com/fortanix/guest-components.git"


def provenance(source, attestation_agent, cdh_oneshot):
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    require(commit == FORTANIX_GUEST_COMPONENTS_COMMIT, "Use the reviewed Fortanix guest-components revision")
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], text=True
    )
    require(not status, "Fortanix guest-components source must be clean")
    aa_version = subprocess.check_output([str(attestation_agent), "--version"], text=True, stderr=subprocess.STDOUT)
    require("CCM_AS" in aa_version, "Fortanix Attestation Agent must include ccm_as")
    cdh_version = subprocess.check_output([str(cdh_oneshot), "--version"], text=True, stderr=subprocess.STDOUT)
    require("ccm_kbc: enabled" in cdh_version, "Fortanix CDH must include ccm_kbc")
    return {
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": commit,
        "source_clean": True,
        "features": ["bin", "ccm_as", "ccm_kbc"],
        "build_environment": "ubuntu-26.04-x86_64",
        "attestation_agent_sha256": digest_file(attestation_agent),
        "cdh_oneshot_sha256": digest_file(cdh_oneshot),
    }

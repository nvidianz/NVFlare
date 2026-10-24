# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Measured-guest Fortanix CCM attestation and DSM resource retrieval."""

import base64
import binascii
import contextlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from ...common.errors import require
from ...common.linux import memory_file, run
from ...common.luks import key_release_token
from ..contracts import FORTANIX_LUKS_TOKEN_ID

RUNTIME = Path("/run/cvm")

FORTANIX_CONTROL_ENVIRONMENT = {
    "AA_KBC_PARAMS",
    "AA_SOCKET",
    "APP_CERT_ALT_NAMES",
    "CCM_KBC_APP_ID",
    "CCM_KBC_DSM_ENDPOINT",
    "KBS_CERT",
    "OCICRYPT_KEYPROVIDER_CONFIG",
    "SKIP_SEALED_SECRET_VERIFICATION",
}


def _quoted(value):
    # Validated endpoints, UUIDs and DNS names still go through a real string
    # encoder instead of being interpolated as TOML syntax.
    return json.dumps(value)


def _config_files(directory, config, socket):
    fortanix = config["fortanix_ccm"]
    aa = directory / "attestation-agent.toml"
    cdh = directory / "cdh.toml"
    aa.write_text(
        "[token_configs.ccm_as]\n"
        + "ccm_domain_names = ["
        + ",".join(_quoted(value) for value in fortanix["ccm_domain_names"])
        + "]\n"
        + f"ccm_appconfig_id = {_quoted(fortanix['ccm_appconfig_id'])}\n\n"
        + "[eventlog_config]\ninit_pcr = 17\nenable_eventlog = false\n\n"
        + '[log]\nlevel = "error"\n'
    )
    cdh.write_text(
        "[kbc]\n"
        + 'name = "ccm_kbc"\n'
        + f"url = {_quoted(fortanix['dsm_endpoint'])}\n\n"
        + "[aa]\n"
        + f"aa_socket = {_quoted('unix://' + str(socket))}\n\n"
        + "[kbc_configs.ccm_kbc]\n"
        + f"dsm_app_id = {_quoted(fortanix['dsm_app_id'])}\n\n"
        + '[log]\nlevel = "error"\n'
    )
    aa.chmod(0o600)
    cdh.chmod(0o600)
    return aa, cdh


@contextlib.contextmanager
def authorized_key(
    config,
    digest,
    token,
    *,
    budget,
    credentials_path=None,
    credentials_state_path=None,
    minimum_credentials_validity=None,
    maximum_credentials_age=None,
):
    """Obtain one fresh CCM certificate and export the bound DSM secret."""
    require(isinstance(digest, bytes) and len(digest) == 32, "Invalid vault binding")
    require(
        key_release_token({"tokens": {FORTANIX_LUKS_TOKEN_ID: token}}) == token,
        "Fortanix vault is missing its authenticated DSM object token",
    )
    require(
        credentials_path is None
        and credentials_state_path is None
        and minimum_credentials_validity is None
        and maximum_credentials_age is None,
        "Application-visible attestation credentials are not supported by the Fortanix runtime",
    )
    require(type(budget) in (int, float) and budget > 0, "Invalid Fortanix attestation budget")
    deadline = time.monotonic() + budget
    RUNTIME.mkdir(mode=0o700, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fortanix-", dir=RUNTIME) as temporary:
        directory = Path(temporary)
        directory.chmod(0o700)
        socket = directory / "attestation-agent.sock"
        aa_config, cdh_config = _config_files(directory, config, socket)
        environment = {key: value for key, value in os.environ.items() if key not in FORTANIX_CONTROL_ENVIRONMENT}
        environment["RUST_LOG"] = "off"
        process = subprocess.Popen(
            [
                config["attestation_agent"],
                "--attestation_sock",
                "unix://" + str(socket),
                "--config",
                str(aa_config),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
        )
        try:
            while not socket.exists():
                require(process.poll() is None, "Fortanix Attestation Agent stopped before becoming ready")
                require(time.monotonic() < deadline, "Fortanix Attestation Agent startup timed out")
                time.sleep(0.05)
            remaining = deadline - time.monotonic()
            require(remaining > 0, "Fortanix attestation exceeded its total time budget")
            uri = "kbs:///dsm/key/" + token["dsm_object_id"]
            encoded = run(
                [
                    config["cdh_oneshot"],
                    "--config",
                    str(cdh_config),
                    "--retry",
                    "0",
                    "get-resource",
                    "--resource-uri",
                    uri,
                ],
                timeout=remaining,
                env=environment,
                operation="Fortanix CCM attestation and DSM resource retrieval",
                secret=True,
            ).strip()
            try:
                key = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error):
                require(False, "Fortanix CDH returned invalid resource encoding")
            require(
                len(key) == 64 and base64.b64encode(key) == encoded, "Fortanix DSM returned an invalid vault secret"
            )
            with memory_file(key) as key_fd:
                yield key_fd
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)

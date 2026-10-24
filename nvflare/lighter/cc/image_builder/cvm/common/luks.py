# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate authenticated LUKS headers, mappings and payload reads."""

import json
import uuid

from .contracts import FORTANIX_LUKS_TOKEN_ID, FORTANIX_LUKS_TOKEN_TYPE, HEADER_BYTES
from .errors import require
from .linux import memory_file, run

FORTANIX_CCM = "fortanix_ccm"

# The unlock secret is 512 random bits, so the keyslot KDF adds no security.
# Pin a cheap deterministic KDF: a benchmarked argon2 cost would make guest
# unlock time and memory depend on the build host and could exhaust guest RAM.
KEYSLOT_KDF = {"type": "pbkdf2", "hash": "sha256", "iterations": 1000}


def key_release_token(metadata):
    """Return the one authenticated provider token, rejecting unknown tokens."""
    tokens = metadata.get("tokens", {})
    require(isinstance(tokens, dict), "Invalid LUKS2 token metadata")
    if not tokens:
        return None
    require(set(tokens) == {FORTANIX_LUKS_TOKEN_ID}, "Vault must contain at most one recognized key-release token")
    token = tokens[FORTANIX_LUKS_TOKEN_ID]
    object_id = token.get("dsm_object_id") if isinstance(token, dict) else None
    try:
        canonical_object_id = isinstance(object_id, str) and str(uuid.UUID(object_id)) == object_id
    except (ValueError, AttributeError):
        canonical_object_id = False
    require(
        isinstance(token, dict)
        and set(token) == {"type", "keyslots", "schema_version", "provider", "dsm_object_id"}
        and token.get("type") == FORTANIX_LUKS_TOKEN_TYPE
        and token.get("keyslots") == ["0"]
        and token.get("schema_version") == 1
        and token.get("provider") == FORTANIX_CCM
        and canonical_object_id,
        "Invalid Fortanix LUKS2 key-release token",
    )
    return token


def install_key_release_token(device, token):
    """Add provider metadata before the authenticated header is frozen."""
    require(key_release_token({"tokens": {FORTANIX_LUKS_TOKEN_ID: token}}) == token, "Invalid key-release token")
    encoded = (json.dumps(token, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with memory_file(encoded) as token_fd:
        run(
            [
                "cryptsetup",
                "token",
                "import",
                "--token-id",
                FORTANIX_LUKS_TOKEN_ID,
                "--json-file",
                f"/proc/self/fd/{token_fd}",
                device,
            ],
            pass_fds=(token_fd,),
            secret=True,
        )
    require(key_release_token(inspect_header(device)) == token, "Fortanix token was not stored in the LUKS2 header")


def validate_luks_metadata(metadata):
    segments = metadata.get("segments", {})
    require(set(segments) == {"0"}, "Vault must contain exactly one encrypted segment")
    segment = segments["0"]
    require(segment.get("type") == "crypt" and segment.get("encryption") == "aes-xts-random", "Wrong vault cipher")
    require(
        segment.get("offset") == str(HEADER_BYTES) and segment.get("sector_size") == 512, "Wrong vault storage layout"
    )
    require(
        segment.get("integrity", {}).get("type") in ("hmac-sha256", "hmac(sha256)"),
        "Vault lacks keyed payload authentication",
    )
    require(not segment.get("flags"), "Unsupported vault segment flags")
    config = metadata.get("config", {})
    require(
        int(config.get("json_size", -1)) == 12288 and int(config.get("keyslots_size", -1)) == HEADER_BYTES - 32768,
        "Unsupported metadata/keyslot area",
    )
    require(not config.get("flags") and not config.get("requirements"), "Unsupported persistent cryptsetup options")
    slots = metadata.get("keyslots", {})
    require(set(slots) == {"0"}, "Vault requires one fixed keyslot")
    slot = slots["0"]
    require(
        slot.get("type") == "luks2" and slot.get("key_size") == 96,
        "Expected 512-bit XTS plus 256-bit HMAC key material",
    )
    kdf = slot.get("kdf", {})
    require(
        all(kdf.get(key) == value for key, value in KEYSLOT_KDF.items()),
        "Vault keyslot must use the pinned deterministic KDF",
    )
    require(
        slot.get("af") == {"type": "luks1", "stripes": 4000, "hash": "sha256"},
        "Vault keyslot must use the pinned anti-forensic splitter",
    )
    area = slot.get("area", {})
    require(
        int(area.get("offset", 0)) >= 32768
        and int(area.get("offset", 0)) + int(area.get("size", HEADER_BYTES)) <= HEADER_BYTES,
        "Keyslot is outside the frozen header",
    )
    key_release_token(metadata)
    return segment


def inspect_header(device, header_fd=None):
    args = ["cryptsetup", "luksDump", "--dump-json-metadata", device]
    if header_fd is not None:
        args += ["--header", f"/proc/self/fd/{header_fd}"]
    metadata = json.loads(run(args, pass_fds=() if header_fd is None else (header_fd,), secret=True))
    validate_luks_metadata(metadata)
    return metadata


def snapshot_header(device):
    with open(device, "rb", buffering=0) as stream:
        data = stream.read(HEADER_BYTES)
    require(len(data) == HEADER_BYTES, "Truncated logical vault header")
    return data


def validate_mapping(mapper):
    """Check the activated targets, never request dmsetup --showkeys."""
    table = run(["dmsetup", "table", mapper]).decode().split()
    require(len(table) >= 9 and table[2] == "crypt", "Expected dm-crypt target")
    require(table[3] == "capi:authenc(hmac(sha256),xts(aes))-random", "Activated authenticated cipher mismatch")
    # cryptsetup 2.x loads the LUKS2 volume key as a kernel logon key, which
    # user space cannot read back. An inline key would be dumpable by root.
    key = table[4].split(":")
    require(
        table[4].startswith(":") and len(key) >= 4 and key[2] == "logon",
        "Vault volume key must be a kernel logon keyring reference",
    )
    require("integrity:48:aead" in table[8:], "Missing authenticated IV/HMAC tags")
    require(not any("allow_discards" in x or "recalculate" in x for x in table), "Unsafe dm-crypt options")
    underlying = table[6]
    require(":" in underlying, "Unexpected dm-integrity device reference")
    major, minor = underlying.split(":")
    integrity = run(["dmsetup", "table", "-j", major, "-m", minor]).decode().split()
    require(len(integrity) >= 8 and integrity[2] == "integrity", "Missing dm-integrity target")
    require(integrity[5] == "48" and integrity[6] == "J", "dm-integrity must use journal mode and 48-byte tags")
    require(
        not any(x in ("recalculate", "reset_recalculate", "allow_discards") for x in integrity),
        "Unsupported dm-integrity options",
    )
    return underlying


def scan(device):
    # read() propagates EIO. A short read is normal only at the device boundary.
    size = int(run(["blockdev", "--getsize64", device]))
    total = 0
    with open(device, "rb", buffering=0) as stream:
        while total < size:
            chunk = stream.read(min(1024 * 1024, size - total))
            require(chunk, "Short authenticated scan")
            total += len(chunk)

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

"""Project configuration discovery, credential paths and acceptance authorities."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cvm.build import config
from cvm.common.errors import BuildError

PUBLIC_KEY = (
    ed25519.Ed25519PrivateKey.generate()
    .public_key()
    .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
)


class ProjectConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.build = self.root / "project/config/generated/vault_build.yml"
        self.build.parent.mkdir(parents=True)
        self.build.write_text("{}\n")

    def write_project(self, directory, hostname="keys.test"):
        directory.mkdir(parents=True, exist_ok=True)
        credentials = directory / "credentials"
        credentials.mkdir(exist_ok=True)
        for name in ("ca", "admin_token_file"):
            (credentials / name).write_text("fixture")
        (credentials / "admin_token_file").chmod(0o600)
        (credentials / "acceptance.pub").write_bytes(PUBLIC_KEY)
        trustee = {"url": "https://" + hostname}
        trustee.update({name: "credentials/" + name for name in ("ca", "admin_token_file")})
        value = {
            "key_release": {"provider": "trustee", "trustee": trustee},
            "approval": {"public_keys": ["credentials/acceptance.pub"]},
        }
        path = directory / "cvm_project.yml"
        path.write_text(yaml.safe_dump(value))
        return path

    def test_ancestor_discovery_and_paths_use_project_not_build_directory(self):
        path = self.write_project(self.root / "project")
        value = config.project(self.build)
        service = value["key_release"]["trustee"]
        self.assertEqual(service["admin_token_file"], str(path.parent / "credentials/admin_token_file"))
        self.assertEqual(service["url"], "https://keys.test")
        self.assertEqual(service["expiry_warning_seconds"], 7 * 86400)
        self.assertEqual(value["approval"]["public_keys"], [str(path.parent / "credentials/acceptance.pub")])

    def test_expiry_warning_window_is_configurable(self):
        path = self.write_project(self.root / "project")
        value = yaml.safe_load(path.read_text())
        value["key_release"]["trustee"]["expiry_warning_seconds"] = 86400
        path.write_text(yaml.safe_dump(value))
        self.assertEqual(config.project(self.build)["key_release"]["trustee"]["expiry_warning_seconds"], 86400)

        for invalid in (-1, 1.5, "86400", True):
            value["key_release"]["trustee"]["expiry_warning_seconds"] = invalid
            path.write_text(yaml.safe_dump(value))
            with self.subTest(invalid=invalid), self.assertRaises(BuildError):
                config.project(self.build)

    def test_legacy_trustee_schema_preserves_expiry_warning_window(self):
        path = self.write_project(self.root / "project")
        value = yaml.safe_load(path.read_text())
        value["trustee"] = value.pop("key_release")["trustee"]
        value["trustee"]["expiry_warning_seconds"] = 86400
        path.write_text(yaml.safe_dump(value))
        release = config.project(self.build)["key_release"]
        self.assertEqual(release["provider"], "trustee")
        self.assertEqual(release["trustee"]["expiry_warning_seconds"], 86400)

    def test_nearest_project_wins_and_invalid_nearest_does_not_fall_back(self):
        self.write_project(self.root / "project", "outer.test")
        nearest = self.write_project(self.build.parent, "nearest.test")
        self.assertEqual(config.project(self.build)["key_release"]["trustee"]["url"], "https://nearest.test")
        nearest.write_text("{}\n")
        with self.assertRaises(BuildError):
            config.project(self.build)

    def test_explicit_project_overrides_discovery_for_staged_yaml(self):
        self.write_project(self.build.parent, "local.test")
        path = self.write_project(self.root / "outside", "selected.test")
        value = config.project(self.build, path)
        self.assertEqual(value["key_release"]["trustee"]["url"], "https://selected.test")
        self.assertEqual(value["key_release"]["trustee"]["ca"], str(path.parent / "credentials/ca"))
        with self.assertRaises(BuildError):
            config.project(self.build, self.root / "missing.yml")

    def test_missing_project_fails(self):
        # Do not depend on whether the workstation has its own ancestor config.
        with (
            patch.object(Path, "exists", return_value=False),
            patch.object(Path, "is_symlink", return_value=False),
            self.assertRaisesRegex(BuildError, "--project-config"),
        ):
            config.project(self.build)

    def test_invalid_project_schema_or_endpoint_is_rejected(self):
        path = self.write_project(self.root / "project")
        valid = yaml.safe_load(path.read_text())
        approval = valid["approval"]
        release = valid["key_release"]
        trustee = release["trustee"]
        for value in (
            {},
            {"key_release": None, "approval": approval},
            {"key_release": release, "approval": approval, "unknown": True},
            {"key_release": {**release, "other": {}}, "approval": approval},
            {"key_release": {"provider": "unknown", "unknown": {}}, "approval": approval},
            {
                "key_release": {"provider": "trustee", "trustee": {**trustee, "token": "unexpected"}},
                "approval": approval,
            },
            {"key_release": {"provider": "trustee", "trustee": {"url": "https://keys.test"}}, "approval": approval},
            {"key_release": release},
            {"key_release": release, "approval": {}},
            {"key_release": release, "approval": {"public_keys": []}},
            {"key_release": release, "approval": {"public_keys": "credentials/acceptance.pub"}},
            {"key_release": release, "approval": {"public_keys": ["credentials/ca"]}},
            {"key_release": release, "approval": {"public_keys": ["credentials/missing.pub"]}},
        ):
            path.write_text(yaml.safe_dump(value))
            with self.subTest(value=value), self.assertRaises(BuildError):
                config.project(self.build)
        for url in (
            None,
            [],
            "http://keys.test",
            "https:///missing-host",
            "https://user:pass@keys.test",
            "https://keys.test?q=1",
        ):
            value = {
                "key_release": {"provider": "trustee", "trustee": {**trustee, "url": url}},
                "approval": approval,
            }
            path.write_text(yaml.safe_dump(value))
            with self.subTest(url=url), self.assertRaises(BuildError):
                config.project(self.build)

    def test_missing_credential_and_duplicate_yaml_key_are_rejected(self):
        path = self.write_project(self.root / "project")
        (path.parent / "credentials/admin_token_file").unlink()
        with self.assertRaises(BuildError):
            config.project(self.build)
        path.write_text("key_release: {}\nkey_release: {}\n")
        with self.assertRaises(BuildError):
            config.project(self.build)

    def test_management_credential_permissions_are_restricted(self):
        path = self.write_project(self.root / "project")
        token = path.parent / "credentials/admin_token_file"
        token.chmod(0o640)
        with self.assertRaisesRegex(BuildError, "group or other"):
            config.project(self.build)

    def test_fortanix_schema_is_strict_and_resolves_credentials(self):
        directory = self.root / "project"
        credentials = directory / "credentials"
        credentials.mkdir(parents=True)
        (credentials / "dsm.key").write_text("fixture")
        (credentials / "dsm.key").chmod(0o600)
        (credentials / "acceptance.pub").write_bytes(PUBLIC_KEY)
        value = {
            "key_release": {
                "provider": "fortanix_ccm",
                "fortanix_ccm": {
                    "dsm": {
                        "endpoint": "https://dsm.test",
                        "account_id": "00000000-0000-4000-8000-000000000001",
                        "group_id": "00000000-0000-4000-8000-000000000002",
                        "provisioning_api_key_file": "credentials/dsm.key",
                    },
                },
            },
            "approval": {"public_keys": ["credentials/acceptance.pub"]},
        }
        path = directory / "cvm_project.yml"
        path.write_text(yaml.safe_dump(value))
        loaded = config.project(self.build)
        settings = loaded["key_release"]["fortanix_ccm"]
        self.assertEqual(settings["dsm"]["provisioning_api_key_file"], str(credentials / "dsm.key"))

        for mutation in (
            lambda v: v["key_release"]["fortanix_ccm"].update(ccm={}),
            lambda v: v["key_release"]["fortanix_ccm"]["dsm"].update(endpoint="http://dsm.test"),
            lambda v: v["key_release"]["fortanix_ccm"]["dsm"].update(account_id="not-a-uuid"),
            lambda v: v["key_release"]["fortanix_ccm"]["dsm"].pop("group_id"),
        ):
            candidate = yaml.safe_load(yaml.safe_dump(value))
            mutation(candidate)
            path.write_text(yaml.safe_dump(candidate))
            with self.subTest(candidate=candidate), self.assertRaises(BuildError):
                config.project(self.build)

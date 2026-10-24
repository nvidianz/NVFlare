# Fortanix CCM/DSM Provider for CVM Builder

**Status:** CPU-only preview implemented with the reviewed Fortanix CoCo guest-components revision; production use requires service qualification and the acceptance matrix below

**Target:** NVFlare 2.9 CVM Builder

**Updated:** 2026-10-24

## 1. Decision

CVM Builder supports two explicit key-release providers:

- `trustee`: CoCo Trustee supplies attestation, policy, reference values, and key resources.
- `fortanix_ccm`: Fortanix CCM supplies workload attestation and Fortanix DSM stores and releases the vault secret.

The Fortanix path does not run Trustee. It uses Fortanix's CoCo integration at [guest-components commit `a77b529`](https://github.com/fortanix/guest-components/commit/a77b52902554946c3cd77166483915c915465895): Attestation Agent `ccm_as` obtains a short-lived CCM workload certificate, and CDH `ccm_kbc` uses that certificate to export the selected DSM `SECRET`. CVM Builder invokes the provider through `cdh-oneshot get-resource`; it does not implement a second Fortanix attestation protocol.

An existing CoCo deployment and CVM Builder may use the same Fortanix account, CCM service, DSM instance, Zone CA, Node Agents, and reviewed guest-component build. Use a dedicated CCM AppConfig, DSM runtime application, DSM group, and provisioning application for each independently administered NVFlare security domain. One project may place all of its CVM vault objects in that group.

Provider selection is measured into the generic CVM profile. The build project and the selected profile must name the same provider, and runtime fallback is forbidden.

## 2. Why this resolves the KBS-interface gap

Fortanix agent check-in is an attestation operation, not a Trustee-compatible KBS HTTP endpoint. The implementation therefore uses the data-plane interface already supplied by the Fortanix CoCo fork:

1. The measured guest starts a private `ttrpc-aa` process configured for `ccm_as`.
2. `ccm_as` collects SNP or TDX evidence and asks the host Node Agent and CCM for a workload certificate whose key is generated in the guest.
3. The guest runs `cdh-oneshot get-resource --resource-uri kbs:///dsm/key/<uuid>` with `ccm_kbc` selected.
4. `ccm_kbc` presents the CCM certificate to DSM and exports that DSM `SECRET`.
5. CVM Builder accepts only the canonical base64 encoding of exactly 64 bytes and passes it to `cryptsetup` through a memory file.

`kbs:///dsm/key/<uuid>` is the CoCo resource URI understood by `ccm_kbc`; no Trustee service is contacted.

## 3. Trust and authorization model

The Fortanix deployment supplies two service identities:

- A **runtime DSM application** authenticated through the CCM Zone CA. Its application ID is measured into the generic CVM configuration and passed to `ccm_kbc`. It may export secrets from the project's DSM group.
- A **Stage 2 provisioning application** whose API key remains on the trusted build host. It may list, import, inspect, update, and delete `SECRET` objects in the same group. It is never copied into a CVM or delivery artifact.

CCM AppConfig policy admits the approved generic CVM measurements and acceptable SNP or TDX TCB. DSM group membership controls which objects that workload identity can export. Separate projects or mutually untrusted tenants require separate AppConfigs, runtime applications, DSM groups, and provisioning credentials.

### 3.1 Binding a DSM object to a vault

A DSM UUID supplied only through an environment variable or host command line would be replaceable by the untrusted host. Stage 2 therefore stores the UUID in a custom LUKS2 token before freezing the header:

```json
{
  "type": "nvflare-fortanix-ccm",
  "keyslots": ["0"],
  "schema_version": 1,
  "provider": "fortanix_ccm",
  "dsm_object_id": "00000000-0000-4000-8000-000000000002"
}
```

The existing vault binding hashes the complete logical LUKS2 header. SNP `HOST_DATA` or TDX `MRCONFIGID` carries that digest. Before any network request, the measured guest compares the attached header with the local TEE report. Changing the DSM UUID changes the header digest and fails this local check. Unknown, additional, or malformed LUKS2 tokens are rejected.

The CCM identity is profile-scoped rather than vault-scoped. Vault isolation comes from all of these controls together:

- the DSM UUID is authenticated by the TEE-bound LUKS2 header;
- only the measured bootstrap reads the token and requests the secret;
- `cryptsetup` accepts the returned bytes only for that header's keyslot;
- the application container has no Attestation Agent socket, CDH binary, TEE device, DSM credential, or vault-key file;
- the opened volume key resides in the kernel logon keyring and is not readable by user space.

A byte-for-byte copy of an authorized vault retains the same UUID, DSM object, and authorization, matching the existing CVM copy semantics.

## 4. One-time Fortanix deployment setup

Before building a profile, the Fortanix operator creates or selects:

1. A CCM AppConfig that admits the exact CPU-only SNP and/or TDX generic CVM profile and acceptable platform TCB.
2. A DSM Trusted-CA runtime application that trusts the CCM Zone CA and is associated with that AppConfig's workload identity.
3. A DSM group containing that runtime application.
4. A separate DSM API-key provisioning application with the minimum object-management operations needed by Stage 2 in that group.
5. A registered Fortanix Node Agent on every physical CVM host.

CVM Builder does not create CCM AppConfigs, DSM applications, groups, or Node Agent registrations. Those long-lived service objects are deployment bootstrap, shared with the site's CoCo infrastructure where the same administrative trust domain permits it.

## 5. Configuration

### 5.1 Stage 1 profile

The generic CVM profile selects Fortanix and pins the exact guest implementation and long-lived service identities:

```yaml
key_release_provider: fortanix_ccm
gpu: none

fortanix_ccm:
  guest_components_commit: a77b52902554946c3cd77166483915c915465895
  dsm_endpoint: https://sdkms.example.org
  dsm_account_id: 00000000-0000-4000-8000-000000000001
  dsm_group_id: 00000000-0000-4000-8000-000000000002
  dsm_app_id: 00000000-0000-4000-8000-000000000003
  ccm_appconfig_id: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  ccm_domain_names:
    - ccm.example.org

platforms:
  amd_sev_snp:
    enabled: true
    attester: snp
    firmware: /usr/share/ovmf/OVMF.amdsev.fd
    cpu_model: EPYC-v4
    attestation_agent: ../inputs/ttrpc-aa
    cdh_oneshot: ../inputs/cdh-oneshot
    guest_components_provenance: ../inputs/fortanix_guest_components_build.json
  intel_tdx:
    enabled: true
    attester: tdx
    firmware: ../inputs/OVMF.inteltdx.fd
    cpu_model: host
    quote_generation: {type: vsock, cid: 2, port: 4050}
    attestation_agent: ../inputs/ttrpc-aa
    cdh_oneshot: ../inputs/cdh-oneshot
    guest_components_provenance: ../inputs/fortanix_guest_components_build.json
```

The provenance record must identify a clean `https://github.com/fortanix/guest-components.git` checkout at the pinned commit, the `ccm_as`, `ccm_kbc`, and `bin` features, an `ubuntu-26.04-x86_64` build environment, and SHA-256 digests for both binaries. Stage 1 copies only those binaries into the measured root.

Generate that record on the Ubuntu 26.04 build host after compiling the pinned checkout. The command checks the revision, clean worktree, and feature strings embedded by both binaries before hashing them:

```sh
./cvmctl fortanix-provenance \
  /path/to/fortanix-guest-components \
  /path/to/ttrpc-aa \
  /path/to/cdh-oneshot \
  inputs/fortanix_guest_components_build.json
```

Fortanix profiles currently reject `gpu: nvidia_cc`. The pinned preview collects NVIDIA evidence, but the required negative GPU appraisal and containment matrix has not been qualified. Application-visible peer attestation credentials are also rejected for this provider; CCM credentials are used only inside the measured key-release transaction.

### 5.2 Stage 2 project

`cvm_project.yml` contains only the DSM provisioning boundary and the normal CVM acceptance keys:

```yaml
key_release:
  provider: fortanix_ccm
  fortanix_ccm:
    dsm:
      endpoint: https://sdkms.example.org
      account_id: 00000000-0000-4000-8000-000000000001
      group_id: 00000000-0000-4000-8000-000000000002
      provisioning_api_key_file: ./inputs/dsm-provisioning-api-key

approval:
  public_keys:
    - ./inputs/acceptance-signing.pub
```

The API-key file must be inaccessible to group and other users. The project endpoint, account, and group must exactly match the measured Stage 1 profile. Configuration and credentials stay on the build host and are excluded from the OCI artifacts.

## 6. Lifecycle

### 6.1 Stage 1

`cvmctl build` validates the pinned source provenance, embeds `ttrpc-aa` and `cdh-oneshot`, records their hashes in the CVM manifest, and measures the Fortanix runtime settings into the shared profile contract. Trustee client binaries, Trustee CA material, and Trustee runtime endpoints are omitted.

Generic bundle approval still proves that the exact root and launch artifacts passed the site's CVM acceptance process. Fortanix CCM policy installation is an operator action performed against those approved measurements.

### 6.2 Stage 2

For each platform copy, `cvmctl vault`:

1. Generates a fresh 64-byte vault secret and formats authenticated LUKS2.
2. Imports the bytes into DSM as an enabled, exportable `SECRET` in the configured group.
3. Writes a non-secret `prepared` receipt to `provisioning.json`.
4. Imports the returned DSM UUID as the fixed LUKS2 token and verifies readback.
5. Freezes the header, derives the SNP/TDX vault binding, and finishes all local validation.
6. Updates DSM custom metadata with the final logical resource path and vault-binding digest, verifies readback, and records an `active` receipt.
7. Packages the delivery without `provisioning.json`, the API key, or secret bytes.

If a failure occurs after DSM import, `provisioning.json` and `build_failure.json` retain the object UUID and group without the secret. Both `prepared` and `active` receipts can be passed to the provider's revoke operation. An ambiguous DSM mutation is never retried automatically.

### 6.3 Boot, periodic authorization, and recovery

The guest first verifies dm-verity, the attached LUKS2 header, and the SNP `HOST_DATA` or TDX `MRCONFIGID` binding. It then starts a fresh private Attestation Agent, obtains one CCM workload certificate, and invokes `cdh-oneshot` for the authenticated DSM UUID. The agent is terminated and its temporary socket and configuration are deleted at the end of that transaction.

Every periodic authorization and CPU-only recovery attempt repeats that complete CCM/DSM transaction. No workload certificate is cached across authorization calls. A denied or unavailable CCM/DSM path follows the same quarantine, vault close, bounded recovery, and forced-poweroff behavior as Trustee.

### 6.4 Revocation

Fence active Stage 2 writers, then delete or disable the exact DSM object recorded in `provisioning.json`. That prevents future unlocks but cannot retract a secret already released to a running CVM. Preserve deletion state across DSM backup recovery. Retiring a Fortanix generic profile additionally disables its CCM AppConfig or DSM runtime application after all associated vaults have been drained.

## 7. Security properties and limits

- No DSM management API key or CCM management credential enters the measured guest, encrypted vault, logs, or delivery.
- The runtime uses the reviewed CoCo/Fortanix `ccm_as` and `ccm_kbc` implementation, not an NVFlare verifier or certificate protocol.
- The DSM object UUID is authenticated by the frozen LUKS2 header and local TEE binding.
- The returned resource must be canonical base64 for exactly 64 bytes.
- A Fortanix profile never falls back to Trustee when CCM, the Node Agent, CDH, or DSM fails.
- The preview depends on an unreleased Fortanix guest-components commit. Production rollout must move to a released CoCo/Fortanix interface or explicitly accept and maintain that pinned fork.
- GPU key release and application-visible peer credentials remain unsupported for `fortanix_ccm`.
- As with the Trustee provider, complete-disk rollback and cloning of an authorized root-and-vault pair remain outside the protection supplied by key release alone.

## 8. Acceptance matrix

Before production use, run the same profile on the intended service deployment and hardware:

- Positive CPU-only SNP and TDX build, seal, first unlock, periodic authorization, restart, and CPU recovery.
- Wrong header, changed DSM UUID token, wrong platform, changed root, debug-enabled TEE, unapproved measurement, and unacceptable TCB denial.
- Expired or revoked CCM credential, disabled AppConfig, removed DSM group membership, disabled/deleted secret, wrong DSM application, and wrong DSM group denial.
- Node Agent, CCM, DSM, and DNS timeout tests with bounded quarantine and no fallback.
- Cross-vault attempts in both directions within the shared project group.
- Stage 2 DSM failure before import, after import, after LUKS token installation, and after metadata activation; verify retained receipts and idempotent cleanup.
- Confirmation that OCI artifacts and logs contain no API key, workload private key, vault secret, or provisioning receipt.
- Coexistence with a CoCo workload using the same Fortanix account and service instances while keeping separate policy and DSM authorization boundaries.

Unit tests cover provider selection, DSM lifecycle receipts, strict project/profile schemas, authenticated LUKS token parsing, fresh AA/CDH invocation, resource URI selection, secret length/encoding, package isolation, and fail-closed dispatch. Live Fortanix service and hardware evidence must be added to [VALIDATION.md](VALIDATION.md) when the matrix is run.

## 9. References

- [Fortanix guest-components commit `a77b529`](https://github.com/fortanix/guest-components/commit/a77b52902554946c3cd77166483915c915465895)
- [Fortanix DSM authentication](https://support.fortanix.com/v1/docs/dsm-authentication)
- [Fortanix CoCo deployment on AMD SEV-SNP](https://support.fortanix.com/docs/deploy-confidential-container-coco-workloads-on-amd-sev-snp-platform)
- [Fortanix CoCo deployment on Intel TDX](https://support.fortanix.com/docs/deploy-confidential-container-coco-workloads-on-intel-tdx-platform)
- [Fortanix DSM API](https://support.fortanix.com/apidocs)

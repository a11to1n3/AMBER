# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.4.x   | Yes       |
| < 0.4   | No        |

Security fixes ship on the latest **0.4.x** line until 1.0.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security reports.

Use one of:

1. **GitHub Security Advisories** (preferred):  
   https://github.com/a11to1n3/AMBER/security/advisories/new  
2. Email the maintainers at **security+ambr@proton.me** with:
   - A description of the issue and impact
   - Steps to reproduce or a proof of concept
   - Affected versions / commit SHAs if known

We aim to acknowledge reports within **7 days** and to provide a status
update within **14 days**. Coordinated disclosure timelines are negotiated
case by case; please allow a reasonable window before public discussion.

## Scope notes

AMBER is a scientific simulation library. Typical issues of interest:

- Unsafe deserialization of untrusted run directories or checkpoints
  (`ParallelRunner` checkpoints are **JSON only**; resume requires
  `trust_checkpoint=True` — never unpickle untrusted files)
- Path traversal or write-outside-destination bugs in `RunResults.save` / `load`
  (manifest commit uses exclusive random temps with `O_NOFOLLOW` / `fsync`)
- Supply-chain concerns in published wheels

Out of scope: model scientific validity, GPU driver bugs, and third-party
dependency CVEs (file those upstream; we track via Dependabot / `pip-audit`).

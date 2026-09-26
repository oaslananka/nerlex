# Security Policy

Nerlex is pre-1.0 and is not yet advertised as production-ready.

## Reporting a vulnerability

Please use GitHub's private security reporting / Security Advisory mechanism for this repository when available. Do not disclose suspected vulnerabilities in a public issue before coordinated review.

Include:
- affected version/commit
- reproduction steps
- expected impact
- minimal proof of concept where appropriate

Do not include real credentials, private customer data, or unrelated secrets.

## Security principles

Nerlex intends to:
- keep secret values out of trace/artifact metadata
- prefer safe portable model formats
- validate artifact integrity
- fail closed on incompatible artifacts/schemas
- keep external network/provider calls explicit
- preserve an auditable decision route and artifact identity

These are design goals under active implementation, not current production guarantees.

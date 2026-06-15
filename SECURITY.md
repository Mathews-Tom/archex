# Security Policy

## Supported versions

Security fixes are accepted for the current minor release line and `main`. Older minor releases are reviewed case by case when a low-risk patch can be backported without changing public behavior.

## Reporting vulnerabilities

Report suspected vulnerabilities privately by opening a GitHub security advisory for this repository or by contacting the maintainer listed on the PyPI project page. Do not file public issues for unpatched vulnerabilities.

Include:

- affected archex version or commit
- operating system and Python version
- exact command, API call, MCP request, or Docker image used
- configuration relevant to indexing, model loading, MCP, or Docker
- minimal reproduction steps
- observed impact and any known workaround

Do not include secrets, private source code, proprietary indexes, or model cache contents unless the maintainer explicitly requests them through a private channel.

## Response expectations

The maintainer aims to acknowledge valid private reports within 5 business days, confirm scope and severity after reproduction, and coordinate a fix timeline based on impact. Critical issues that expose secrets, execute untrusted code, or corrupt source/index data are prioritized over hardening requests.

## Scope

In scope:

- archex CLI commands
- Python API entry points
- stdio MCP server behavior
- Docker images published for archex
- model-loading paths and local model cache handling
- repo-local `.archex/` index, vector, graph, and settings state
- dependency and model supply-chain risks that affect normal archex operation

Out of scope:

- social engineering, phishing, or physical attacks
- denial-of-service against public services not operated by archex
- vulnerabilities in downstream LLM clients that consume archex output
- vulnerabilities requiring malicious local administrator access
- reports based only on unsupported forks or modified packages

## Telemetry and data handling

archex does not send telemetry from core CLI, Python API, MCP, or Docker slim workflows. It reads the repository paths you point it at and writes generated state under repo-local `.archex/` or configured cache directories.

Core operation does not require hosted inference credentials or API keys. Optional integrations may require their own client configuration, but archex does not need hosted API keys for core retrieval, indexing, MCP, or Docker slim use.

## Secrets

Do not store secrets in `.archex/` settings, indexes, benchmark manifests, Docker command lines, or MCP client configs. archex indexes repository files; if secrets are present in indexed source files, they can appear in local indexes and returned context bundles. Rotate leaked secrets outside archex and rebuild the affected index after removing them from source.

## Model downloads and remote code

Core BM25-only operation does not load Hugging Face remote code. Optional model paths that require Hugging Face remote code must be explicitly enabled by configuration or command choice, and built-in paths with published revision pins should keep those pins in place.

Model downloads are local to the executing machine or container. Review model licenses, revisions, and cache locations before enabling optional vector, SPLADE, rerank, or remote-code model paths in an enterprise environment.

## Coordinated disclosure

Keep vulnerability details private until a fix or mitigation is available and published. The maintainer will credit reporters who want attribution, unless disclosure would expose sensitive operational details or the reporter requests anonymity.

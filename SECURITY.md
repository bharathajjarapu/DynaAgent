# Security Policy

## Supported Versions

Only the latest commit on `main` is supported.

## Reporting a Vulnerability

Report privately via GitHub: **Security → Report a vulnerability** on [bharathajjarapu/DynaAgent](https://github.com/bharathajjarapu/DynaAgent/security/advisories/new). Do not open public issues for vulnerabilities.

Include affected file/commit, reproduction steps and impact. Expect acknowledgement within 7 days.

## Threat Model

DynaAgent executes **LLM-generated code**, and its inputs (web search results, API responses) are untrusted, so prompt injection can steer what code is generated.

| Asset | Control |
|---|---|
| API keys (`LLM_KEY`, `OPENAI_API_KEY`) | never passed to generated tools; child env is `{PATH}` plus one declared non-secret key |
| Host process | generated code runs in a separate `python -I` interpreter, never `exec` in-process |
| CPU / memory | `RLIMIT_CPU` 15 s, `RLIMIT_AS` 1 GiB, 25 s wall-clock timeout |
| Arithmetic | `calculator` walks a whitelisted AST; no `eval`, bounded exponent and result size |
| Cost | per-query budget on calls, tokens, time and USD |

## Known Limitations

- Process isolation only: generated code can read files and reach the network as the running user.
- Saved tools in `dynaagent.db` are re-executed in later sessions; delete the file to purge them.

**In scope:** sandbox escape, secret leakage, budget bypass, unbounded loops.
**Out of scope:** issues requiring a malicious `.env` or local filesystem access.

For untrusted or multi-user deployments, run DynaAgent inside a container or VM (e.g. Docker with `--network` restrictions, gVisor, nsjail) under an unprivileged user.

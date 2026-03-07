# DynaAgent

**A Dynamic Multi-Agent Generation Framework for Adaptive Complex Reasoning**

Manmath Nath Das, Ajjarapu Bharath, Allaboina Abhishek, Harsh Bhaskerwar
Department of AI & Data Science, VNR Vignana Jyothi Institute of Engineering & Technology, Hyderabad

> A coordinator LLM decomposes a query into a task DAG, synthesizes task-specific agents and tools on demand, executes them in parallel, aggregates candidate answers by voting, and refines until convergence under a hard resource budget.

**Keywords:** dynamic agent generation · multi-agent systems · task decomposition · LLM orchestration · budget-aware convergence

---

## 1. Formulation

An agent is a persona with a tool set, $a = (p, T)$. A dynamic multi-agent system is

$$\mathrm{DMAS} = \big(A(t),\ D(t),\ M(t),\ \phi\big), \qquad \phi : (Q, D(t), M(t)) \to A(t+1)$$

where $A(t)$ is the active agent set, $D(t) = (V(t), E(t))$ the task DAG, $M(t)$ the shared memory and $\phi$ the generation function. Given query $Q$, the objective is a solution $S$ maximizing quality subject to budget $B$.

## 2. Method

| Phase | Operation |
|---|---|
| Analyze | $\pi(Q) \to$ direct answer, or $D = (V, E)$ with $\lvert V \rvert \le B_{agents}$ and tool specs $\mathcal{T}_{new}$ |
| Generate tools | $\forall\, \tau \in \mathcal{T}_{new}$: synthesize code, validate AST, register a sandboxed callable |
| Generate agents | reuse $a$ if $a.\mathrm{name} = v.\mathrm{agent}$ with $T_a \leftarrow T_a \cup T_v$, else create $a = (p_v, T_v)$ |
| Execute | $v$ starts once $\forall (u, v) \in E: u \in \mathrm{Done}$; ready tasks run concurrently |
| Vote | $k \ge 2$ agents propose answers; plurality consensus (§4) |
| Refine | reviewer returns $(S_r, s_r, \Delta V)$; $D \leftarrow D \cup \Delta V$ |

Latency follows the critical path, not $\lvert V \rvert$: $\ T_{exec} \approx \max_{P} \sum_{v \in P} t_v$ over DAG paths $P$.

## 3. Budget and Convergence

Hard limits, checked before every model call and retry:

$$B = (B_{cost}, B_{time}, B_{agents}, B_{calls}, B_{tokens}) = (\$0.5,\ 300\,\mathrm{s},\ 8,\ 60,\ 5{\times}10^5)$$

$$\mathrm{halt} \iff \exists\, k : u_k \ge B_k \quad\Rightarrow\quad \text{return best partial } S$$

Round $r$ yields reviewer score $s_r \in [0, 10]$. Refinement stops when

$$s_r \ge \tau \ \lor\ s_r - s_{r-1} < \theta \ \lor\ r = R \ \lor\ \Delta V = \varnothing, \qquad \tau = 8,\ \theta = 1,\ R = 3$$

Each agent turn is capped at 6 model steps, the last with tools disabled, so every loop is bounded.

## 4. Voting

Agents $a_1, \dots, a_k$ propose answers $c_1, \dots, c_k$; each agent $a_j$ casts ballot $b_j \in \{1, \dots, k\}$:

$$n_i = \sum_{j} \mathbb{1}[b_j = i], \qquad i^\star = \arg\max_i n_i \quad (\text{ties} \to \text{earliest ballot})$$

| Mode (`DYNA_VOTE`) | Authors visible | Candidate order | Casting |
|---|---|---|---|
| `anonymous` (default) | no | random permutation $\sigma_j$ per voter | parallel |
| `identified` | yes | fixed, live tally shown | sequential |
| `off` | – | – | – |

Logged per query to study transparency and order effects (self-vote rate, first-vote win):

$$\mathrm{SVR} = \frac{1}{k} \sum_{j} \mathbb{1}[b_j = j], \qquad \mathrm{FVW} = \mathbb{1}[b_1 = i^\star]$$

## 5. Memory

| Level | Scope | Store |
|---|---|---|
| $M_{short}$ | query | outputs propagated along DAG edges |
| $M_{mid}$ | session | last 4 dialogue turns |
| $M_{long}$ | persistent | SQLite: plans with $s \ge \tau$, recalled by FTS5 BM25 (top 2, $s \ge 7$); validated tools |

Generated tools persist only after their first successful call.

## 6. Tools and Isolation

- `calculator`: AST-whitelisted arithmetic (no `eval`), bounded exponent and result size.
- `search`: provider-hosted web search.
- Generated tools: stdlib-only Python in an isolated interpreter (`python -I`), $\mathrm{env} = \{\mathrm{PATH}\}$, `RLIMIT_CPU` 15 s, `RLIMIT_AS` 1 GiB, 25 s wall clock. See [SECURITY.md](SECURITY.md).

## 7. Setup

Python 3.14 with [uv](https://docs.astral.sh/uv/). The core has zero third-party dependencies (stdlib client for any OpenAI-compatible Responses API); Streamlit is an optional extra.

```bash
uv sync --extra ui
uv run main.py                              # CLI
uv run --extra ui streamlit run app.py      # web UI
```

| `.env` variable | Default | Purpose |
|---|---|---|
| `LLM_KEY` / `OPENAI_API_KEY` | – | API key |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Responses API endpoint |
| `LLM_MODEL` | `gpt-5-mini` | worker model |
| `LLM_SMART_MODEL` | `LLM_MODEL` | coordinator and `smart` tasks |
| `DYNA_VOTE` | `anonymous` | voting mode |

## 8. Evaluation

| Benchmark | Task | Metric |
|---|---|---|
| GPQA-Diamond | graduate-level science MCQ | accuracy |
| MuSR | multi-step soft reasoning | accuracy |
| IFEval | verifiable instruction following | official checker |

```bash
uv run eval.py musr --limit 30              # DynaAgent
uv run eval.py musr --limit 30 --single     # single-model baseline
uv run eval.py gpqa --limit 30              # gated: requires HF_TOKEN
uv run eval.py ifeval --limit 50            # responses for the IFEval scorer
```

Reports accuracy, SVR, FVW, tokens, calls, cost and latency; per-item JSONL in `results/`.

## 9. Structure

```
core.py     coordinator: planning, DAG scheduler, agents, voting, review; SQLite memory
llm.py      Responses API client, bounded tool loop, budget
tools.py    calculator, web search, sandbox, tool validation
prompts.py  prompts and strict JSON schemas
eval.py     GPQA-Diamond, MuSR, IFEval harness
main.py     CLI
app.py      Streamlit UI
```

## Citation

```bibtex
@misc{das2026dynaagent,
  title  = {DynaAgent: A Dynamic Multi-Agent Generation Framework for Adaptive Complex Reasoning},
  author = {Das, Manmath Nath and Bharath, Ajjarapu and Abhishek, Allaboina and Bhaskerwar, Harsh},
  year   = {2026},
  note   = {VNR Vignana Jyothi Institute of Engineering \& Technology}
}
```

## License

Proprietary, all rights reserved; see [LICENSE](LICENSE). Also: [Security](SECURITY.md) · [Code of Conduct](CODE_OF_CONDUCT.md).

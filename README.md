# packlight

Travel light. packlight aims to give your AI agent only the skills, tools and context each
request needs, and keep the rest out of the way.

**Status: early (v0.0.1).** Only the registry scanner works today. The routing is not built yet,
and nothing here has been benchmarked. Please don't depend on it yet.

## The problem

Agents list or load every skill and tool up front. That costs tokens on every session, the model
often picks the wrong one or ignores the right one, and long sessions keep re-reading context
that no longer matters.

## The idea

- **Capabilities are data, not something the model memorises.** Each request is scored against
  the current list of skills, tools and context chunks (name, description, start of the body).
  The model learns the general task "does this request need this capability?". Adding a skill
  means rescanning, not retraining.
- **Any agent, any scorer.** A small core (command line and Python library) plus adapters: a
  Claude Code hook, an MCP server, and others later. The scorer is swappable: keyword search
  (runs anywhere), embeddings, or a small fine-tuned decision model.
- **Local.** Prompts never leave your machine. Any released model is trained only on public or
  synthetic data.

## What works today

```bash
python -m packlight.registry
```

Scans skills (Claude Code, Codex, Hermes, the shared `~/.agents/skills`, plugins), agents, MCP
servers and context files. It merges duplicates, estimates token costs, and writes
`data/registry.json`. The folders it reads are built-in defaults for now. A config file is
planned.

## How we will decide it is good enough

It is only switched on in real sessions if it passes all of this on skills it never saw in
training:

| Check | Bar |
|---|---|
| Needed skill is in the top 3 | at least 95% |
| Prompts that need a skill but get none | at most 2% |
| Prompts that need no skill but get one | at most 10% |
| Gap between unseen and seen skills | at most 3 points |
| Latency on a CPU | under 150 ms |
| Skill tokens per replayed session | at least 30% fewer, same task success |

It must also beat these baselines: load everything, keyword search, embeddings, embeddings plus
rerank, and the skill selection in [Ratel](https://github.com/ratel-ai/ratel) and
[hermes-jev-skills](https://github.com/kerpopule/hermes-jev-skills). It then runs in shadow mode
(logging its picks, injecting nothing) for at least two weeks before anything is injected.

## Roadmap

1. Registry (done).
2. Baselines and a public, reproducible test set with held-out skills.
3. Training data from public and synthetic sources.
4. A fine-tuned scorer, compared against the best baseline.
5. Shadow mode, then promotion only if the bar above is met.
6. Adapters for Claude Code, MCP and other agents.

## Related work

Ratel (tool and skill catalogs), hermes-jev-skills (decision-model skill selection),
laya-conductor (Laya fine-tuned for routing), and the SkillRet, SkillFlow and SkillSeek papers
on skill retrieval. This project does not claim to be first. It aims to be portable and
measured.

## License

Apache-2.0. See [LICENSE](LICENSE).

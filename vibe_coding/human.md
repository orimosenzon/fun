# The Human's Guide

*How humans contribute to MerchSage.*

---

`doc/CONTRIBUTING.md` is the agent-facing API reference to the codebase. This document is for you, the human. Your job is to decide what should exist, set the standards, and improve the system that agents use to build it.

---

## Principles

**1. You decide, agents execute.** You set direction, priorities, and quality standards. Agents write code, sync it, test it, and iterate.

**2. Fix friction at the root.** When an agent struggles — wrong patterns, missing context, repeated questions — fix it systemically. Update a doc, add a coding standard, build a skill. One fix applies to every future session.

**3. Observe agent work.** When an agent takes a wrong turn, ask why. Unclear docs? Missing convention? Thin context? Every friction point is a signal about what to improve.

**4. Start with conversation.** Describe the problem before prescribing the solution. Let the agent explore the code and surface constraints. "I'm thinking about..." beats "Implement X."

**5. Docs are the interface.** In an agent-native codebase, documentation programs agent behavior. A coding standard prevents a category of bugs. A doc update changes how every future session works.

**6. Automate repetition.** A skill is a markdown file. A hook is a shell script. If you type the same instructions twice, capture it. Use `/sk-skill` to create new skills from friction.

**7. The agent finishes the job.** The agent syncs code, tests it, reads error logs, fixes what's broken, and tries again — until it works. You review after it reports success, not after it writes the first draft.

**8. Be specific.** Vague instructions produce vague code. State constraints, provide context, or better yet put it in the docs so the agent always has it.

---

## How It Works

MerchSage is built on Claude Code with skills, hooks, an MCP server, and persistent memory.

### Skills

Skills are slash commands that load context or execute workflows. `/sk-main` loads the pipeline architecture docs. `/sk-test` detects what changed, syncs, runs the test pipeline, and reports results.

There are 23 skills covering pipeline stages, frontends, git workflows, testing, documentation, and server provisioning. See [skills.md](arch/skills.md) for the full list.

### Compact Hooks

When a session gets long, Claude Code compacts earlier conversation. This loses the skill docs loaded at the start. Two shell hooks fix this:

- **Before compaction**: detects the active skill, saves it to a temp file
- **After compaction**: tells the agent to re-read all relevant docs

Automatic — no intervention needed.

### MCP Server

A custom MCP server gives the agent direct access to Kestra (the pipeline orchestrator). The agent can sync code, trigger runs, check status, read logs, and run smoke tests — without shell commands. This is what makes autonomous test-fix-retry loops practical.

### Memory

Persistent memory file (`~/.claude/projects/.../memory/MEMORY.md`) survives across sessions. Contains workarounds, connection details, and patterns learned from past sessions.

### Teams (experimental)

Infrastructure for spawning parallel agents — a lead writing code, a doc agent updating docs, a commit agent making clean commits. Still being refined.

---

## Your Workflow

### Starting a session

1. Open Claude Code
2. Load the relevant skill: `/sk-main`, `/sk-shopify`, `/sk-mockups`, etc.
3. Describe the problem or idea. Let the agent explore and ask questions.
4. Once aligned, let the agent work through to completion.

### When you see friction

- **Agent asks a question it shouldn't need to ask** → update the docs
- **Agent writes code that doesn't match your patterns** → add a coding standard
- **You keep typing the same setup instructions** → build a skill with `/sk-skill`
- **Agent loses important context across sessions** → add it to memory
- **A multi-step workflow keeps being error-prone** → build a workflow skill

### Building a new skill

Use `/sk-skill`. It interviews you about the friction or idea, checks existing skills for overlap, and produces a ready-to-use SKILL.md. Skills are markdown files in `.claude/skills/` — no build step, no deployment.

### Reviewing agent work

The agent handles its own testing. Your review happens after it reports success:

- **Check the approach**: Did it solve the right problem?
- **Scan for standards violations**: Does it match your patterns?
- **Look at the edges**: Error handling, missing validations, security
- **Spot-check the output**: Look at what the pipeline actually produced

Be specific about what's wrong. "This doesn't handle the case where X is empty" is useful. "This doesn't look right" is not.
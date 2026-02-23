---
name: explore-github-repo
description: Use this skill when asked to explore, understand, or answer questions about a GitHub repository given a GitHub URL
---

You will be given a GitHub URL, for example:

- `https://github.com/karpathy/nanochat`
- `https://github.com/pytorch/pytorch/issues/12345`
- `https://github.com/owner/repo/pull/42`

The user may also ask a specific question about the repo alongside the URL.

### Part 1: Parse the URL

Extract the `{owner}` and `{repo}` from the URL. Also note if the URL points to a specific resource (issue, PR, file, etc.).

### Part 2: Get repo overview

Use both tools in parallel:

1. **DeepWiki MCP** — Call `read_wiki_structure` with the repo `{owner}/{repo}` to get the documentation topic tree. This gives you a high-level map of the repo's architecture and components.

2. **GitHub CLI** — Run `gh repo view {owner}/{repo}` to get the repo description, stars, language, and other metadata.

### Part 3: Answer the user's question

Use the right tool for the job:

**For understanding architecture, how things work, or "explain X":**
- Use DeepWiki `read_wiki_contents` to read specific documentation pages from the topic tree
- Use DeepWiki `ask_question` to ask targeted questions about the repo

**For concrete repo data (issues, PRs, code, releases, stats):**
- `gh issue list -R {owner}/{repo}` — List open issues
- `gh issue view {number} -R {owner}/{repo}` — View a specific issue
- `gh pr list -R {owner}/{repo}` — List open PRs
- `gh pr view {number} -R {owner}/{repo}` — View a specific PR (add `--comments` for discussion)
- `gh api repos/{owner}/{repo}/contents/{path}` — View file contents
- `gh release list -R {owner}/{repo}` — List releases
- `gh api repos/{owner}/{repo}` — Raw API for anything else

**For searching code or issues:**
- `gh search code "{query}" --repo {owner}/{repo}` — Search code
- `gh search issues "{query}" --repo {owner}/{repo}` — Search issues

If the user didn't ask a specific question, provide a concise overview: what the repo does, its architecture, key components, and recent activity.

### Part 4: Follow up

After answering, ask the user if they want to explore anything else about the repo. You already have context — you can dive deeper into specific topics, issues, PRs, or code paths as needed.

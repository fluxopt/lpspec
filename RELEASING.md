# Releasing

The git tag is the version. `pyproject.toml` carries no version number —
hatch-vcs derives it from the tag at build time, so nothing can drift and a
release can be cut from any branch without editing a file first.

One publish path: **pushing a `v*` tag ships it.** Every route below just
produces a tag; [`publish.yaml`](.github/workflows/publish.yaml) does the rest
(test → build → verify the built version matches the tag → GitHub release).

## Where this project is

**Below 1.0.0, and not by accident.** A breaking change bumps the minor, a
feature and a fix bump the patch, and reaching 1.0.0 takes an edit to the
config.

| you want | you do | version you get |
| --- | --- | --- |
| day-to-day development | nothing | `0.0.1.dev22+ged5056087` — hatch-vcs numbers every commit |
| a release someone can pin | merge the release PR | `0.0.2`, `0.0.3`, … |
| the minor to move | land a breaking marker | `0.1.0` |
| a cut from another branch | run **Prerelease** | a named stream like `0.2.0rc1` |
| to reach 1.0.0 | edit the config on purpose (below) | `1.0.0` |

Untagged commits are already uniquely versioned and installable, so there is no
reason to tag until someone needs a fixed reference. A cut from the
**Prerelease** workflow still needs `--prerelease` to resolve, so a prerelease
cannot be picked up by accident.

## Normal release

Land conventional commits on `main`.
[`release.yaml`](.github/workflows/release.yaml) keeps a release PR open with
the computed version and changelog. Merge it → release-please tags `v0.0.2` →
publish runs → dist version `0.0.2`. Which commit types
appear is `changelog-sections` in
[`.release-please-config.json`](.release-please-config.json); `chore`, `test`,
`ci`, `build` and `style` are hidden.

**How the version moves.** Two keys decide it, and both stay until 1.0.0.
`bump-patch-for-minor-pre-major` sends a `feat:` down the *patch* row, and
`bump-minor-pre-major` sends a breaking change down the *minor* row rather than
reaching for 1.0.0. So below 1.0.0 the minor says a consumer has to change
something, and the patch says everything else.

A breaking marker is how you ask for the minor: a `!` in the subject, or a
`BREAKING CHANGE:` footer. Both were refused while the version sat on
`0.0.1-alpha.N`, because there a marker moved the base version off the stream —
the accident at #251, where a `feat!:` on `0.0.1-alpha.12` produced
`0.1.0-alpha.12`. There is no stream to move off now, so the marker is ordinary
input and [`pr-title.yaml`](.github/workflows/pr-title.yaml) no longer looks for
it.

**The subject that lands on main.** `main` takes squash merges only, so one PR
is one commit and its subject is what release-please parses — the rule a
contributor has to follow, and the allowed types, are in
[CONTRIBUTING.md](CONTRIBUTING.md#branches-commits-prs). What matters here is
the failure mode: a subject release-please cannot parse breaks nothing, it just
goes silently missing from the changelog. That silence is why
[`pr-title.yaml`](.github/workflows/pr-title.yaml) is a *required* check rather
than advisory.

Because `squash_merge_commit_title` is `COMMIT_OR_PR_TITLE`, GitHub uses the PR
title on a multi-commit PR and the commit's own title on a single-commit one;
the check validates both, so neither can slip through.

## Branch protection

`main` is covered by a repository ruleset: no deletion, no force-push,
squash-only merges through a PR, and two required checks — `ci` (from
[`ci.yml`](.github/workflows/ci.yml)) and `Conventional commit subject`. The CI
job has a fixed name and no matrix, so the ruleset names it directly and never
needs updating; keep it that way. If a Python matrix ever comes back, put an
aggregating gate job in front of it rather than naming legs — requiring
`full (3.11)` would mean adding a version leaves it unrequired, and dropping
one blocks every PR on a check that can no longer report.

A required check must exist on `main` before it is required — land the workflow
first, then add it to the ruleset. Approvals are not required (solo repo); a
review count of 0 still forces the PR, the squash and the checks.

### Relaxed while in early development

Actions bills per job, rounded up to the minute, and the suite takes ~5s. So CI
cost is job count, and while nothing downstream pins a release — no users to
break, runner minutes the scarcer resource — it is one job
that deliberately trades coverage for cost. What that gives up:

- **Only Python 3.11 is tested.** The 3.12 and 3.13 classifiers in
  `pyproject.toml` are untested claims. 3.11 is the floor, so it catches the
  common breakage (reaching for a newer stdlib feature) but not the reverse: a
  removal or deprecation that only bites on 3.13.
- **Only two dependency sets are installed:** current-with-dev, and the declared
  floors bare. The floors are exercised *without* linopy/xarray, so the linopy
  lane is only ever tested against current linopy — narrow, since the lane
  resolves to one branch anyway (the `[linopy]` extra's direct reference,
  pending the v1 release).

This list used to carry a third entry, and it is worth keeping the correction
rather than the claim: *"the PR-title check does not re-run on `synchronize` …
the cost is a missing CHANGELOG line, not a broken release."* The cost was a
**permanently blocked PR**. A required check is evaluated against the head
commit, so skipping the push event leaves the new head with no result at all,
and GitHub waits for one that never arrives — every visible check green, merge
blocked, nothing to click. #269 sat like that for hours and was reported as CI
hanging. The trigger is back; the saving was one ~5s job per push.

Tighten these before anyone downstream pins a release — that is the point where
a missed regression reaches somebody rather than just us. A Python matrix is the
first thing to add back, behind a gate job. Until then, prefer spending minutes
on the suite over spending them on matrix breadth.

## Overriding the version

Three levers, ascending force:

1. **Edit the release PR** before merging — retitle it; release-please follows
   the PR, not just the commits.
2. **`Release-As:` footer** on any commit forces the next version:
   `git commit --allow-empty -m "chore: release 0.3.0" -m "Release-As: 0.3.0"`
3. **Tag by hand** — `git tag -a v0.3.0 -m v0.3.0 && git push origin v0.3.0`
   publishes immediately, bypassing release-please. The changelog will not
   mention it, so keep this for emergencies.

## Prereleases

Run the **Prerelease** workflow from any branch (Actions → Prerelease → Run
workflow). It computes the next counter, runs lint and the suite, and pushes the
tag; `dry-run` prints it without pushing. Name the version the cut leads to
(`0.2.0`) and pick the channel; counters are tracked per version and channel.
There is no default version, because release-please no longer cuts a stream for
one to match.

Tags are dashed semver (`v0.2.0-rc.1`), which normalises to the PEP 440 dist
version `0.2.0rc1`. Do **not** hand-tag `v0.2.0rc1` — without the dash, publish
marks the GitHub release as a full release.

**On `main`, prefer the release PR.** release-please cuts no prereleases, so the
two no longer count into one namespace — but this workflow reads existing tags
while release-please reads
[`.release-please-manifest.json`](.release-please-manifest.json), and a cut
naming a version release-please has not reached is a tag nobody asked for. Use
**Prerelease** for what it is good at — a cut from a non-`main` branch, or a
differently-named stream.

## Reaching 1.0.0

Drop `bump-minor-pre-major` and `bump-patch-for-minor-pre-major` from
[`.release-please-config.json`](.release-please-config.json). A breaking change
then bumps the major and a feature the minor, which is ordinary semver. Do it in
the pull request that argues the API is stable, because the promise cannot be
withdrawn afterwards.

To keep releasing 0.1.x after `main` moves to 0.2, cut a `0.1.x` branch and run
**Release** with `target-branch: 0.1.x`.

## Consuming an unreleased branch

Don't cut a release for this — install from the ref:

```bash
pixi add --pypi "specsolve @ git+ssh://git@github.com/fluxopt/specsolve@feat/some-branch"
pixi add --pypi "specsolve @ git+https://github.com/fluxopt/specsolve@d09aab6"
```

Every tagged build also attaches its wheel and sdist to the GitHub release.

## One-time setup

- **Release app** — a GitHub App with `contents: write` + `pull-requests: write`,
  credentials in secrets `APP_CLIENT_ID` / `APP_PRIVATE_KEY`. Needed so release
  PRs run CI and prerelease tags trigger publish. Without it, `release.yaml`
  degrades to `GITHUB_TOKEN` and warns; `prerelease.yaml` refuses to run. In
  place: release PRs are authored by `fluxopt-release-bot[bot]`.
- **PyPI** — currently off. The `pypi` job is skipped unless the repo variable
  `PUBLISH_TO_PYPI` is `true`. To go live: register a
  [trusted publisher](https://docs.pypi.org/trusted-publishers/) for
  `specsolve` (workflow `publish.yaml`, environment `pypi`), create the `pypi`
  environment, then set the variable.

  **PyPI refuses a direct reference, and this package declares two.** Both have
  to go before one upload succeeds, and they clear in a fixed order.

  The `linopy` extra is `linopy @ git+…@master`, because the arithmetic
  convention that lane requires is in no linopy release. It becomes an ordinary
  floor when upstream ships the release that carries it
  ([#463](https://github.com/fluxopt/specsolve/issues/463)); `pyproject.toml`
  says why no floor stands in for it meanwhile.

  The runtime dependency `math-spec @ git+…` waits on nothing but order.
  math-spec publishes, and the pin becomes a floor on the version it published.
  It cannot become a floor sooner: the `floors` environment resolves the
  declared dependencies from PyPI, so `test-floors` fails on a version that is
  not there yet.

  Everything else — the tag, the wheel, the GitHub release — works today.

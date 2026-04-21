# Kotlin Parity Release Checklist

Use this checklist before making a Kotlin parity claim in a release PR, changelog, or tag announcement.

Release version: `<vX.Y.Z>`
Release date (UTC): `<YYYY-MM-DD>`
Owner: `<name>`

## 1. Test Evidence Links (Required)

- [ ] Test command and result summary are attached.
  - Command: `./gradlew test`
  - Optional focused commands:
    - `./gradlew test --tests 'ragas.BackendsTest' --tests 'ragas.PublicApiTest' --tests 'ragas.cli.CliParityTest'`
  - Evidence links/artifacts:
    - `build/reports/tests/test/index.html`
    - `build/test-results/test/`
- [ ] Core parity suites verified (link to runs or report snippets):
  - [ ] evaluation/runtime parity (`src/test/kotlin/ragas/EvaluationTest.kt`, `src/test/kotlin/ragas/EvaluationParityHooksTest.kt`)
  - [ ] metric fixture parity (`src/test/kotlin/ragas/GoldenFixturesTest.kt`, `src/test/kotlin/ragas/metrics/collections/*FixtureTest.kt`)
  - [ ] backend/discovery parity (`src/test/kotlin/ragas/BackendsTest.kt`)
  - [ ] CLI parity (`src/test/kotlin/ragas/cli/CliParityTest.kt`)
  - [ ] public API stability smoke (`src/test/kotlin/ragas/PublicApiTest.kt`)
- [ ] Any failing/skipped suites are explicitly listed with rationale and follow-up issue links.

## 2. Deferred-Scope Review (Required)

Review and confirm that release notes clearly separate shipped parity from intentional deferrals.

- [ ] Deferral list reviewed against current `Plan.md` and `PARITY_MATRIX.md`.
- [ ] For each deferral below, capture rationale, risk, owner, and next review milestone.

| Deferral Area | Rationale | User Risk | Owner | Next Review |
| --- | --- | --- | --- | --- |
| Multimodal ingestion hardening | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| Full WS6 synthesis breadth | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| Broader integrations | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| Bundled Google Drive backend | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| Exact Python DSPy internals | `<fill>` | `<fill>` | `<fill>` | `<fill>` |
| Full Python CLI UX breadth | `<fill>` | `<fill>` | `<fill>` | `<fill>` |

## 3. Versioned Doc-Freeze Gates (Required)

Complete these gates before tagging.

### Gate A: Release-Candidate Freeze

- [ ] `Plan.md`, `PARITY_MATRIX.md`, `README.md`, `MIGRATION.md`, and `API_SURFACE.md` are updated for this version.
- [ ] `Last updated` dates and WS status markers are consistent across docs.
- [ ] Parity claim wording matches implemented/tested scope only.

### Gate B: Tag Gate

- [ ] Final release notes reference this completed checklist and test evidence links.
- [ ] No unresolved contradictions between:
  - `Plan.md` WS statuses
  - `PARITY_MATRIX.md` status rows
  - documented intentional deferrals
- [ ] Git tag/version in build metadata matches release notes.

## 4. Parity-Claim Statement (Required)

Use this template in release notes:

> Kotlin parity claim for `<vX.Y.Z>` is based on the completed parity checklist,
> attached test evidence, and explicit deferred-scope disclosure.

- [ ] Statement included in release notes/changelog.

## 5. Sign-off

- [ ] Engineering sign-off: `<name/date>`
- [ ] QA/verification sign-off: `<name/date>`
- [ ] Docs sign-off: `<name/date>`

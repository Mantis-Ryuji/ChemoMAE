# ChemoMAE Backlog

Track unfinished work here. Keep completed release history in `CHANGELOG.md`
and Git. Add accepted fixes and improvements with GitHub issue links when
available.

## Publication and repository links

- [ ] Add the manuscript/publication link to `README.md`'s Research background
  section when its public URL is available. Keep the link as optional research
  background for the general-purpose library.

## v0.2.4 publication

- [ ] Confirm CI passes for the release commit.
- [ ] Publish `v0.2.4` through the tag-triggered PyPI workflow.
- [ ] Verify the published package metadata, README rendering, versioned
  documentation link, and installed-package CPU smoke test in a clean environment.

## Clustering follow-up

- [ ] Low-priority consideration: a label-only `VMFMixture.predict` path that
  avoids the full N-by-K responsibility matrix. Preserve labels, tie behavior,
  and chunk semantics if this path is changed. Caller-side slicing is documented
  as the current alternative.

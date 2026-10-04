# ChemoMAE Backlog

Track unfinished work here. Keep completed release history in `CHANGELOG.md`
and Git. Add accepted fixes and improvements with GitHub issue links when
available.

## Publication and repository links

- [ ] Add the manuscript/publication link to `README.md`'s Research background
  section when its public URL is available. Keep the link as optional research
  background for the general-purpose library.

## Clustering follow-up

- [ ] Low-priority consideration: a label-only `VMFMixture.predict` path that
  avoids the full N-by-K responsibility matrix. Preserve labels, tie behavior,
  and chunk semantics if this path is changed. Caller-side slicing is documented
  as the current alternative.

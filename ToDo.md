# ChemoMAE Backlog

Track unfinished work here. Keep completed release history in `CHANGELOG.md`
and Git. Add accepted fixes and improvements with GitHub issue links when
available.

## Clustering follow-up

- [ ] Low-priority consideration: a label-only `VMFMixture.predict` path that
  avoids the full N-by-K responsibility matrix. Preserve labels, tie behavior,
  and chunk semantics if this path is changed. Caller-side slicing is documented
  as the current alternative.

## Sampling follow-up

- [ ] Add count-based downsampling to `CosineFPS` alongside the existing
  `ratio`-based option.

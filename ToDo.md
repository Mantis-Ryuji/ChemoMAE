# ChemoMAE Backlog

Track unfinished work here. Keep completed release history in `CHANGELOG.md`
and Git. Add accepted fixes and improvements with GitHub issue links when
available.

## Publication and repository links

- [ ] Add the manuscript/publication link to `README.md`'s Research background
  section when its public URL is available. Keep the link as optional research
  background for the general-purpose library.

## v0.2.4 release preparation

The source version is 0.2.4; publication is deferred. The elbow score-direction
fix, selected-artifact return path, dtype diagnostics, and usage guides are
implemented. User-reported targeted/full tests and the selected documentation
recipes have passed in `chemomae-test`. The wheel and source distribution have
been built and passed `twine check`. The selected documentation recipes also
passed against installed ChemoMAE 0.2.4 in `chemomae-min` with Torch 2.1.0 on
CPU. The dedicated installed-package CPU smoke check and `pip check` also
passed in that environment; see `CHANGELOG.md` for the recorded results.
The planned local validation is complete. Publication preparation remains below.

- [ ] Before an explicitly authorized publication, complete the same-commit CI
  gates, update preparation-only installation/status wording, and pin the
  package documentation URL to the confirmed v0.2.4 snapshot. Rebuild the wheel
  and source distribution after the final documentation edits, then rerun
  `twine check`.

## Clustering follow-up

- [ ] Low-priority consideration: a label-only `VMFMixture.predict` path that
  avoids the full N-by-K responsibility matrix. The current need is limited;
  implementation is not a v0.2.4 release requirement. Preserve labels, tie
  behavior, and chunk semantics if this path is changed. Caller-side slicing is
  documented as the current alternative.

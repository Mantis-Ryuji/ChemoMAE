# ChemoMAE Backlog

Track unfinished work here. Keep completed release history in `CHANGELOG.md`
and Git. Add accepted fixes and improvements with GitHub issue links when
available.

## Documentation review

- [ ] User: review `README.md` and all of `docs/` for wording, scientific
  explanations, example clarity, links, and GitHub MathJax rendering.
- [ ] Verify the revised formulas in GitHub's rendered Markdown or unsaved
  Preview. MathJax command support and source syntax have been checked, but
  the changed pages still need actual GitHub display verification. CPU example
  execution and local link checks do not establish rendering correctness.

Keep this review open until the user confirms completion. Add corrections
identified during the review as individual tasks and remove them once resolved.

## Publication and repository links

- [ ] Add the manuscript/publication and research-repository links to
  `README.md`'s Research background section when their public URLs are available.
  Use confirmed URLs and keep the links as optional research background for the
  general-purpose library.

## Clustering

- [ ] Fix the score-direction mismatch in `elbow_vmf` before relying on its
  returned elbow K. `src/chemomae/clustering/vmf_mixture.py` negates BIC/mean NLL
  before passing the curve to `find_elbow_curvature`, which enforces a
  nonincreasing curve with `np.minimum.accumulate`. For monotonically decreasing
  scores such as `[500, 410, 360, 340, 335, 334]` at K=1..6, the negated curve
  becomes constant after that operation, discarding the bend and returning
  K=2 with zero curvature. Confirm the intended treatment of decreasing and
  nonmonotonic BIC/NLL curves; add a regression check that covers score
  direction, not only return types/ranges; then synchronize the API docs.
  This documentation revision records the limitation without changing the
  algorithm or choosing a new model-selection rule.

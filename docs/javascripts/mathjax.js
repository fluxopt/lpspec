// pymdownx.arithmatex in `generic: true` mode rewrites `$...$` / `$$...$$`
// into `\(...\)` / `\[...\]` before MathJax sees the page, so the escaped
// delimiters are the ones that matter. The dollar forms are enabled too: the
// typeset output this site quotes is written with them, and a block pasted out
// of `to_markdown` should render the same here as it does on GitHub.
//
// There is deliberately no `ignoreHtmlClass`. What such a guard would protect
// is covered anyway: MathJax's default `skipHtmlTags` includes `pre` and
// `code`, so a `$` in a fenced block or an inline span is never scanned and the
// delimiters only ever meet prose.
window.MathJax = {
  tex: {
    inlineMath: [
      ["\\(", "\\)"],
      ["$", "$"],
    ],
    displayMath: [
      ["\\[", "\\]"],
      ["$$", "$$"],
    ],
    processEscapes: true,
    processEnvironments: true,
  },
};

// Instant navigation swaps the document without a reload, so MathJax has to be
// told to run again. `document$` is the builder's own observable, and it emits
// on every swap as well as on the first load. Without this the equations on a
// page reached by a click reach the reader as literal text: they are in the
// DOM as `.arithmatex` spans and MathJax never sees them.
//
// The three resets before the typeset pass are what makes a second visit to a
// page render the same as the first. `typesetClear` drops the nodes MathJax is
// tracking from the outgoing page, `texReset` restarts equation numbering, and
// `clearCache` drops the font metrics measured against it.
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});

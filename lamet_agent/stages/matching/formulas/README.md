# Precomputed matching formulas

Each `*.md` here is the `## Matching Formula` body for one kernel: the closed-form
matching coefficient plus the code-vs-paper consistency check, generated once by the
LLM that read `kernels.py` next to the kernel's arXiv paper. Shipping them means the
matching report costs no arXiv download and no ~27k-token prompt at run time.

The file name is `<kernel_id>.<language>.md` -- one kernel, one file, always. The
header carries a `digest=` over the kernel's own source together with its
`@kernel_reference` tag. Edit a kernel by one sign and the digest stops matching, so
the stale text is not served: the formula is regenerated, re-cross-checked against the
paper, and written back over the same file.

Do not edit these by hand. Regenerate after adding or changing a kernel:

```
lamet-agent precompute-formulas --backend api --model deepseek/deepseek-chat --prune
```

Anything not covered here is generated at run time into `~/.cache/lamet-agent/formulas`
(override or disable with `LAMET_FORMULA_CACHE_DIR`).

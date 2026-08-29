Request manual user confirmation only after every validator Issue is resolved.
Summarize the accepted physics and configuration changes in natural language for
a physicist; never provide a code diff or imply that Run has already begun.

Call schema: `summary` is a required concise string describing the complete Plan.
`changes` is a required array of natural-language changes meaningful to the user.
The observation reports `ready=true` only when validation currently succeeds.

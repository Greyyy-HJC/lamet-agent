Find candidate files and directories below the manifest directory or configured
project root. Use path discovery whenever a required path is missing or ambiguous;
do not guess a path from its likely name.

Call schema: `query` is a required search string. `max_results` is an optional
integer from 1 through 100 and defaults to 30. The observation returns matching
paths ordered by the Plan state path finder.

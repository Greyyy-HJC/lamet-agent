Undo the most recent successful manifest update. A natural-language request such
as “undo that change” must call this tool immediately rather than being interpreted
as a new manifest value.

Call schema: this tool takes no arguments; call it with `{}`. The observation
reports whether anything was undone and returns the refreshed validator Issues.

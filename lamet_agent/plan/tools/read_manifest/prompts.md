Read the current in-memory authored manifest, or one subtree, when the user asks
to see or understand current settings. Use it for informational questions as well
as edits, and never infer a current value from stale conversation text.

Call schema: `path` is an optional JSON Pointer string. Omit it or use an empty
string to read the complete manifest. The observation returns the resolved path
and current JSON value, or an error when the path does not exist.

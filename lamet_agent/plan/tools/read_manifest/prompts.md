Read the current in-memory authored manifest, or one subtree, when the user asks
to see current settings. Never infer a current value from stale conversation text.

Call schema: `path` is an optional JSON Pointer string. Omit it or use an empty
string to read the complete manifest. The observation returns the resolved path
and current JSON value, or an error when the path does not exist.

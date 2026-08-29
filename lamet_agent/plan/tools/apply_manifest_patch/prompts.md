Apply guarded RFC 6902 changes to the in-memory authored candidate only after the
user's answer establishes the intended values. Never claim success until the
observation reports `ok=true`; it also returns a fresh, lossless validator result.

Call schema: `patches` is a required nonempty array. Each item contains `op`
(`add`, `replace`, or `remove`), a JSON Pointer `path`, and a `value` for operations
that write data. Use `add` for missing keys, `replace` for existing keys, and
`remove` only for invalid or explicitly rejected settings.

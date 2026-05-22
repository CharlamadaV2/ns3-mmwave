# src/util

@brief Low-level string, path, and INI parsing utilities shared across all modules.  \

Performs reads run.ini into a two-level map (section → key → value)
that ConfigLoader then queries to populate the SimConfig.

Contains helpers for trimming whitespace, splitting on tabs,
formatting timestamps as ISO-8601, resolving relative paths against a base
directory, and parsing comma-separated seed lists.

## Output

All functions return values to their callers.


## Module Layout

| File | Description |
|------|-------------|
| `ini-parser.h` | `IniMap` type alias and declarations for `parseIni`, `iniGet`, `iniGetBool`. |
| `ini-parser.cc` | Line-by-line INI parser: comment stripping, section headers, key-value splitting. |
| `string-utils.h` | Declarations for `trimStr`, `splitTab`, `toIso8601`, `resolvePath`, `dirOf`, `parseSeedList`. |
| `string-utils.cc` | Implementations of all string and path utility functions. |



## Conventions

- **`parseIni` strips both `#` and `;` comments.** Both comment styles are
  supported on the same line. Everything from the first `#` or `;` to the end
  of the line is discarded before any other processing.

- **Keys before any `[section]` header are stored under the empty string key.**
  `ini[""]` holds any key-value pairs that appear before the first section
  header in the file. In practice `run.ini` always starts with a section, so
  this is rarely relevant.

- **`iniGetBool` treats `"true"`, `"1"`, and `"yes"` as true (case-insensitive).**
  Everything else — including `"false"`, `"0"`, and `"no"` — is false. There
  is no error for unrecognised values; they silently return `false`.

- **`toIso8601` uses `gmtime_r` (POSIX only).** The output is always UTC,
  formatted as `"YYYY-MM-DDTHH:MM:SSZ"`. Sub-second precision is discarded.
  This function is not portable to Windows without a compatibility shim.

- **`resolvePath` returns the path unchanged if empty or absolute.** It never
  throws — an empty path in, an empty path out. Callers that need to validate
  existence must do so separately.



## Dependencies

No Dependency
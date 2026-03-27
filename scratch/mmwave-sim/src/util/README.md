# util/

Generic helpers with no simulation or ns-3 knowledge. Stdlib only.

## Files

### ini-parser.h / ini-parser.cc

INI-style configuration file parser.

**Key types:**
- `IniMap` -- `map<string, map<string, string>>` (section -> key -> value)

**Key functions:**
- `parseIni(path)` -- reads an INI file, returns `IniMap`. Supports `[section]` headers, `key=value` pairs, `#` and `;` comments.
- `iniGet(ini, section, key, default)` -- retrieve a string value with fallback
- `iniGetBool(ini, section, key, default)` -- retrieve a boolean value with fallback

### string-utils.h / string-utils.cc

String manipulation and path utilities.

**Key functions:**
- `trimStr(s)` -- trim leading/trailing whitespace
- `splitTab(line)` -- split by tab character
- `toIso8601(time_point)` -- format `system_clock::time_point` as ISO 8601 UTC string
- `resolvePath(base_dir, path)` -- resolve a relative path against a base directory
- `dirOf(path)` -- return parent directory of a path

## Usage

Used by `config/` for INI parsing and path resolution, and by `io/` for trace file parsing and timestamp formatting.

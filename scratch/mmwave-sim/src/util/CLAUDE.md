# util/

## Scope
Generic helpers with no simulation or ns-3 knowledge.

## Files
- **ini-parser.h/cc** -- `parseIni()`, `iniGet()`, `iniGetBool()` for INI config file reading
- **string-utils.h/cc** -- `trimStr()`, `splitTab()`, `toIso8601()`, `resolvePath()`, `dirOf()`

## Dependencies
- Depends on: nothing (stdlib only)
- Depended on by: `config/`, `io/`

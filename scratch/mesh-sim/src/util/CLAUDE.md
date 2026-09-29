# util/

## Scope
Generic string, path, and INI helpers. Standard library only: no ns-3, no `domain/` types.

## Files
- **ini-parser.h/cc** -- `IniMap`, `parseIni()`, `iniGet()`, `iniGetBool()`
- **string-utils.h/cc** -- `trimStr()`, `splitTab()`, `toIso8601()`, `resolvePath()`, `dirOf()`, `parseSeedList()`

## Behavior to preserve
- `parseIni` strips `#` and `;` comments before parsing, so values cannot contain
  either character. It throws `std::runtime_error` if the file cannot be opened.
- `iniGetBool` is true only for `true`/`1`/`yes` (case-insensitive); any other
  present value is false, with no error.
- `parseSeedList` calls `std::exit(1)` on a bad token; it does not throw.
- `toIso8601` uses POSIX `gmtime_r` and always outputs UTC.
- `resolvePath` returns empty or absolute inputs unchanged.
- `splitTab` has no callers in `src/` today.

## Changing a helper
Both `.cc` files are listed in the root `CMakeLists.txt` and in
`tests/unit/config/Makefile`; a new `.cc` here must be added to both. There is
no `util/` test suite of its own; `make -C tests/unit/config test` compiles
these files and exercises `parseSeedList`.

## Dependencies
- Depends on: standard library only
- Depended on by: `config/` (`config-loader.cc`, `rl-control.cc`), `cli/` (`cli-parser.cc`), `io/` (`metrics-writer.cc`, `run-logger.h`), `tests/unit/config`

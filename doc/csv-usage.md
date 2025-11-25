# CSV Lua Library – Function Reference

This document describes the public functions of the CSV Lua module.  
It assumes the file is saved as `csv.lua` and required as:

```lua
local csv = require("csv")
```

The library exposes:

- **Module-level helpers** such as `csv.rows`, `csv.read_all`, `csv.count_rows`, …
- **Reader instances** created with `csv.new(opts)`, which expose richer stateful methods
  like `set_excluded`, `fit_normalizer`, `min_max`, etc.

All functions below are described in terms of **what they do**, their **arguments**, and
their **return values**.

---

## Common concepts

Many functions take a `source` argument. It can be one of:

1. **Filename (string)** – path to a CSV file (opened with `io.open(source, "r")`).
2. **Open file handle** – any object that supports `:read("*l")` and optionally `:close()`.
3. **Raw CSV string** – a string that is *not* the path to an existing file.

Other shared behaviors:

- **Header mode (`header` option)**  
  - When `header = true`, the first non-comment row is treated as a header row.  
    All subsequent rows are returned as **tables indexed by column name**.  
  - When `header = false`, rows are returned as **array-like tables** (`row[1]`, `row[2]`, …).
- **Type inference (`infer` option)**  
  If enabled, strings that look like numbers are converted to Lua `number`s.
- **Missing values (`na` option)**  
  Field values equal to any entry in the `na` list are converted to `nil`.
- **Streaming**  
  Functions based on `:rows(source)` read the file sequentially; only one row is held
  in memory at a time.

---

## Module-level API

These functions are accessed directly from the module (e.g. `csv.read_all(...)`).
They are thin wrappers that create a new reader internally with the given `opts`.

### `csv.new(opts)`

Create a new CSV reader instance.

```lua
local reader = csv.new{
  header  = true,
  sep     = ",",
  comment = "#",
}
```

**Arguments**

- `opts` (table, optional) – configuration for the reader:
  - `sep` / `delimiter` (string, default `","`)  
    Single-character field separator.
  - `quote` (string, default `"""`)  
    Single-character quote used around fields.
  - `header` (boolean, default `false`)  
    Whether the first non-comment record is a header row.
  - `trim` (boolean, default `true`)  
    If `true`, trims leading/trailing whitespace from unquoted fields.
  - `infer` (boolean, default `true`)  
    If `true`, converts numeric-looking strings to Lua numbers.
  - `comment` (string, optional)  
    If set, any line whose **first non-whitespace character** equals this value
    is treated as a comment and skipped.
  - `na` (table of strings, default `{ "", "NA", "N/A", "NaN", "nan", "null", "NULL" }`)  
    Values equal to any of these are converted to `nil`.
  - `lazy_quotes` (boolean, default `true`)  
    Allows quote characters inside unquoted fields; parsing is more permissive.

**Returns**

- A **reader instance**, referred to as `reader` in the rest of this document.

---

### `csv.rows(source, opts)`

Convenience function to stream rows using a one-off reader.

```lua
for row in csv.rows("data.csv", { header = true }) do
  print(row.Name, row.Age)
end
```

**Arguments**

- `source` – CSV source (see **Common concepts**).
- `opts` (table, optional) – same as for `csv.new`.

**Returns**

- `iter, state` – an iterator function and an internal state object.
  The idiomatic way is to use it directly in a `for` loop:

  ```lua
  for row in csv.rows("data.csv", { header = true }) do
    -- use row
  end
  ```

> Note: Because this function creates a new reader every time, it does **not**
> allow configuring row/column exclusion or normalization via instance methods.
> Use `csv.new` for more complex flows.

---

### `csv.read_all(source, opts)`

Read the entire CSV into memory as a Lua table of rows.

```lua
local rows = csv.read_all("data.csv", { header = true })
print("Number of rows:", #rows)
```

**Arguments**

- `source` – CSV source.
- `opts` (table, optional) – same as for `csv.new`.

**Returns**

- `rows` (table) – array of rows:
  - If `header = true`, each row is a table keyed by column name.
  - If `header = false`, each row is an array indexed by column position.

> Be careful with very large files: `read_all` loads all rows into memory.

---

### `csv.count_rows(source, opts)`

Count the number of data rows in a CSV source.

```lua
local n = csv.count_rows("data.csv", { header = true })
print("Rows:", n)
```

**Arguments**

- `source` – CSV source.
- `opts` (table, optional) – same as for `csv.new`.

**Returns**

- `nrows` (integer) – number of rows **after** applying header handling and
  skipping comment/empty records.

---

### `csv.headers(source, opts)`

Read and return the header row as a list of column names.

```lua
local hdr = csv.headers("data.csv", { header = true })
if hdr then
  for i, name in ipairs(hdr) do
    print(i, name)
  end
end
```

**Arguments**

- `source` – CSV source.
- `opts` (table, optional) – must include `header = true` to get a header.

**Returns**

- `headers` (table or `nil`) – array of column names, or `nil` if `header` is false.

---

### `csv.peek(source, n, opts)`

Read the first `n` rows from the CSV without loading everything.

```lua
local first5 = csv.peek("data.csv", 5, { header = true })
for i, row in ipairs(first5) do
  print(i, row.Name, row.Age)
end
```

**Arguments**

- `source` – CSV source.
- `n` (integer, optional, default `5`) – number of rows to read.
- `opts` (table, optional) – same as for `csv.new`.

**Returns**

- `rows` (table) – up to `n` rows, in the same format as `read_all`
  (keyed by name when `header = true`, otherwise arrays).

---

### `csv.shape(source, opts)`

Return the number of rows and columns in the CSV.

```lua
local nrows, ncols = csv.shape("data.csv", { header = true })
print("Rows:", nrows, "Columns:", ncols)
```

**Arguments**

- `source` – CSV source.
- `opts` (table, optional) – same as for `csv.new`.

**Returns**

- `nrows` (integer) – number of rows (after comment skipping and any implicit handling).
- `ncols` (integer) – number of columns observed in the first data row:
  - If `header = true`, counts named fields in the first returned data row.
  - If `header = false`, counts elements in the first row array.

---

## Reader instance methods

The following methods are available on a reader created via `csv.new(opts)`.

```lua
local reader = csv.new{ header = true }
```

Unless otherwise noted, these methods affect subsequent calls to
`reader:rows(source)` and the convenience methods (`count_rows`, `min_max`, etc.)
on that same reader.

---

### `reader:rows(source)`

Create a streaming iterator over the rows in `source`.

```lua
local reader = csv.new{ header = true }

for row in reader:rows("data.csv") do
  print(row.Name, row.Age)
end
```

**Arguments**

- `source` – CSV source.

**Returns**

- `iter, state`:
  - `iter` – iterator function returning one row at a time (or `nil` at EOF).
  - `state` – small object exposing `state.close()` to manually close the
    underlying file handle early, if desired.

You normally use it directly in a generic `for` loop:

```lua
local iter, state = reader:rows("data.csv")
for row in iter do
  -- process row
end
if state and state.close then
  state.close()
end
```

**Row format**

- If `reader.header == true`:
  - First non-comment record is read as the header.
  - Each subsequent row is returned as a table keyed by header name:
    `row["Name"]`, `row["Age"]`, etc.
- If `reader.header == false`:
  - Each row is returned as an array: `row[1]`, `row[2]`, …

**Parsing behavior**

- Uses RFC4180-style parsing:
  - The separator is `reader.sep`.
  - Quoted fields can contain separators and newlines.
  - Escaped quotes inside quoted fields are written as `""`.
- Respects `reader.trim`, `reader.infer`, `reader.na`, `reader.comment`,
  and any configured row/column exclusions and normalization (see below).

---

### `reader:set_excluded(names, opts)`

Configure **row exclusion** based on a single column.

```lua
local reader = csv.new{ header = true }

reader:set_excluded({ "Alice", "Bob" }, {
  key              = "Name",
  case_insensitive = true,
})
```

Rows where the target column’s value matches one of the `names` will be **skipped**
by subsequent calls to `reader:rows`, `reader:count_rows`, `reader:min_max`, etc.

**Arguments**

- `names` (table) – list of values to exclude (e.g. `{"Alice", "Bob"}`).
- `opts` (table, optional):
  - `key` or `col` – which column to check:
    - When `header = true`: set `key` to a column name (e.g. `"Name"`).
    - When `header = false`: set `col` to a numeric index (e.g. `1`).
  - `case_insensitive` (boolean, default `false`) – if `true`, both the
    column value and the `names` list are compared case-insensitively.

If neither `key` nor `col` is provided:

- When `header = true`, the default column name is `"name"` (case-sensitive).
- When `header = false`, the default column index is `1`.

**Returns**

- The same reader instance (for chaining).

---

### `reader:clear_excluded()`

Remove any previously configured row exclusion.

```lua
reader:clear_excluded()
```

**Arguments**

- None.

**Returns**

- The same reader instance.

After calling this, all rows will be visible again (subject only to other filters
such as column exclusion or normalization).

---

### `reader:set_excluded_columns(names, opts)`

Configure **column exclusion** by header name.

```lua
local reader = csv.new{ header = true }

reader:set_excluded_columns({ "Name", "Email" }, {
  case_insensitive = true,
})
```

Columns listed here will be removed from the rows returned by `reader:rows`.

**Arguments**

- `names` (table) – list of **header names** to drop.
- `opts` (table, optional):
  - `case_insensitive` (boolean, default `false`) – if `true`, header matching
    is done case-insensitively.

**Returns**

- The same reader instance.

**Notes**

- Column exclusion is applied only when `header = true` (because it relies
  on header names). When `header = false`, this setting has no effect.
- Excluded columns do not appear in the returned row tables, and are not
  counted in `reader:shape`.

---

### `reader:clear_excluded_columns()`

Remove any configured column exclusion.

```lua
reader:clear_excluded_columns()
```

**Arguments**

- None.

**Returns**

- The same reader instance.

Subsequent `rows` calls will once again include all columns.

---

### `reader:fit_normalizer(source, columns, opts)`

Compute normalization statistics for selected numeric columns, based on
data from `source`. The normalization is then applied **on the fly** when
rows are read via `reader:rows`.

```lua
local reader = csv.new{ header = true }

-- Fit z-score normalizer on Age and Score columns
reader:fit_normalizer("data.csv", { "Age", "Score" }, {
  method           = "zscore",   -- or "minmax"
  case_insensitive = true,
})

-- Now rows will yield normalized Age and Score
for row in reader:rows("data.csv") do
  print(row.Age, row.Score)
end
```

**Arguments**

- `source` – CSV source from which to compute statistics.
- `columns` (table or `nil`) – which columns to normalize:
  - If `header = true`:
    - Provide a list of column names, e.g. `{ "Age", "Score" }`.
    - If `columns == nil`, numeric columns are auto-detected from the first row.
  - If `header = false`:
    - Provide a list of numeric indices, e.g. `{ 1, 3, 5 }`.
    - If `columns == nil`, numeric indices are auto-detected from the first row.
- `opts` (table, optional):
  - `method` (string, `"zscore"` or `"minmax"`, default `"zscore"`):
    - `"zscore"` – each value becomes  
      `(x - mean) / max(std, eps)`.
    - `"minmax"` – each value becomes  
      `(x - min) / max(max - min, eps)`.
  - `case_insensitive` (boolean, default `false`) – only relevant when using
    column names; enables case-insensitive matching.
  - `eps` (number, default `1e-12`) – small positive constant used when the
    standard deviation or range is zero to avoid division-by-zero.

**Returns**

- The same reader instance.

**Effect on subsequent reads**

After calling `fit_normalizer`, any call to `reader:rows(source2)` (or methods
that use it internally) will:

- For each selected column:
  - Replace numeric values with their normalized value according to the
    fitted statistics.
- Leave all non-selected or non-numeric columns unchanged.

To stop applying normalization, use `reader:clear_normalizer()`.

---

### `reader:clear_normalizer()`

Clear any normalization previously fitted with `fit_normalizer`.

```lua
reader:clear_normalizer()
```

**Arguments**

- None.

**Returns**

- The same reader instance.

After this call, rows will be returned in their original (non-normalized) form.

---

### `reader:min_max(source)`

Compute per-column minimum and maximum values across all rows from `source`.

```lua
local reader = csv.new{ header = true }

local mins, maxs = reader:min_max("data.csv")

print("Age range:", mins.Age, maxs.Age)
print("Score range:", mins.Score, maxs.Score)
```

**Arguments**

- `source` – CSV source.

**Returns**

- `mins`, `maxs` – two tables with the same column layout:
  - If `header = true`:
    - `mins[colname]` and `maxs[colname]` for numeric columns.
  - If `header = false`:
    - `mins[index]` and `maxs[index]` for numeric columns.

Only numeric values (after type inference and any normalization) are considered.

> Note: If a normalizer is configured, `min_max` operates on the **normalized**
> values as they are read.

---

### `reader:count_rows(source)`

Count the number of rows in `source` using this reader’s configuration
(header handling, row exclusion, etc.).

```lua
local reader = csv.new{ header = true }
reader:set_excluded({ "Alice" }, { key = "Name" })

local n = reader:count_rows("data.csv")
print("Rows (excluding Alice):", n)
```

**Arguments**

- `source` – CSV source.

**Returns**

- `nrows` (integer) – number of rows seen by this reader after applying
  header parsing, comment skipping, and row exclusion.

---

### `reader:headers(source)`

Read and return the header row from `source` as an array of column names.

```lua
local reader = csv.new{ header = true }

local hdr = reader:headers("data.csv")
if hdr then
  for i, name in ipairs(hdr) do
    print(i, name)
  end
end
```

**Arguments**

- `source` – CSV source.

**Returns**

- `headers` (table or `nil`) – array of column names, or `nil` if the reader
  is not configured with `header = true`.

This method reads only the first (non-comment) record and does not apply
row/column exclusion or normalization.

---

### `reader:peek(source, n)`

Read the first `n` rows from `source` using this reader’s configuration.

```lua
local reader = csv.new{ header = true }

local sample = reader:peek("data.csv", 3)
for i, row in ipairs(sample) do
  print(i, row.Name, row.Age)
end
```

**Arguments**

- `source` – CSV source.
- `n` (integer, optional, default `5`) – number of rows to read.

**Returns**

- `rows` (table) – up to `n` rows, in the same shape as returned by
  `reader:rows`:
  - If `header = true`, each row is keyed by column name.
  - If `header = false`, each row is an array.

Row exclusion, column exclusion, and normalization are respected.

---

### `reader:shape(source)`

Return the number of rows and columns in `source` as seen by this reader.

```lua
local reader = csv.new{ header = true }

reader:set_excluded_columns({ "Description", "Email" })

local nrows, ncols = reader:shape("data.csv")
print("Rows:", nrows, "Columns (after dropping):", ncols)
```

**Arguments**

- `source` – CSV source.

**Returns**

- `nrows` (integer) – number of rows (after applying row exclusion, comment
  skipping, etc.).
- `ncols` (integer) – number of columns in the first data row **after**
  applying column exclusion:
  - If `header = true`, counts the number of keys in the first row table.
  - If `header = false`, counts the number of elements in the first row array.

---

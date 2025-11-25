------------------------------------------------------------
-- Init
-- See doc/csv-usage.md for instructions.
------------------------------------------------------------

local CSV = {}
CSV.__index = CSV

------------------------------------------------------------
-- Constructor
------------------------------------------------------------
function CSV.new(opts)
    opts = opts or {}
    local self = setmetatable({
        sep           = opts.sep or opts.delimiter or ',',  -- single-char separator
        quote         = opts.quote or '"',
        header        = opts.header or false,
        trim          = opts.trim ~= false,                 -- default true
        infer         = opts.infer ~= false,                -- default true (numbers -> number)
        comment       = opts.comment,                       -- e.g '#'
        na            = opts.na or {"", "NA", "N/A", "NaN", "nan", "null", "NULL"},
        lazy_quotes   = opts.lazy_quotes ~= false,          -- allow quotes inside unquoted fields
        _bom_stripped = false,
        -- Exclusion state
        _exclude          = nil,   -- set of values to exclude (rows)
        _exclude_col      = nil,   -- string header name or numeric index
        _exclude_ci       = false, -- case-insensitive row exclusion
        _drop_cols        = nil,   -- set of header names to drop
        _drop_cols_ci     = false, -- case-insensitive column exclusion
        _drop_cols_idx    = nil,   -- computed indices to drop after reading header
        -- Normalization state
        _norm             = nil,   -- { method='zscore'|'minmax', eps, by_name={}, by_index={}, ci }
    }, CSV)

    assert(#self.sep == 1, 'sep/delimiter must be a single character')
    assert(#self.quote == 1, 'quote must be a single character')
    return self
end

------------------------------------------------------------
-- Utility
------------------------------------------------------------
local function list_to_set(t, to_lower)
    local s = {}
    if to_lower then
        for _, v in ipairs(t) do s[tostring(v):lower()] = true end
    else
        for _, v in ipairs(t) do s[v] = true end
    end
    return s
end

local function make_string_reader(str)
    local i, n = 1, #str
    local reader = {}
    function reader:read(mode)
        if mode ~= '*l' then return nil, 'unsupported read mode' end
        if i > n then return nil end
        local s, e = string.find(str, '?', i)
        local line
        if s then
            line = string.sub(str, i, s - 1)
            i = e + 1
        else
            line = string.sub(str, i)
            i = n + 1
        end
        return line
    end
    function reader:close() end
    return reader
end

------------------------------------------------------------
-- Internal
------------------------------------------------------------
function CSV:_is_comment_line(line)
    if not self.comment or self.comment == '' then return false end
    local first = line:match('^%s*(.)')
    return first == self.comment
end

local UTF8_BOM = string.char(0xEF, 0xBB, 0xBF)
function CSV:_strip_bom_once(buf)
    if self._bom_stripped then return buf end
    self._bom_stripped = true
    if buf and buf:sub(1, 3) == UTF8_BOM then
        return buf:sub(4)
    end
    return buf
end

function CSV:_record_complete(buf)
    if not buf or buf:find(self.quote, 1, true) == nil then return true end
    local doubled = self.quote .. self.quote
    local collapsed = buf:gsub(doubled, '')
    local _, q = collapsed:gsub(self.quote, '')
    return (q % 2) == 0
end

function CSV:_read_record(handle)
    local line = handle:read('*l')
    if not line then return nil end
    line = self:_strip_bom_once(line)
    while self:_is_comment_line(line) do
        line = handle:read('*l')
        if not line then return nil end
    end

    local buf = line
    while not self:_record_complete(buf) do
        local nxt = handle:read('*l')
        if not nxt then break end
        buf = buf .. '' .. nxt
    end
    return buf
end

local function trim(s)
    return s:match('^%s*(.-)%s*$')
end

function CSV:_parse_record(rec)
    local sep   = self.sep
    local quote = self.quote
    local out, field = {}, {}
    local i, n = 1, #rec
    local in_quotes = false

    local function push_field()
        local s = table.concat(field)
        if self.trim and not in_quotes then s = trim(s) end
        table.insert(out, s)
        field = {}
    end

    while i <= n do
        local c = rec:sub(i, i)
        if c == quote then
        if in_quotes then
            local nxt = rec:sub(i + 1, i + 1)
            if nxt == quote then
                table.insert(field, quote)
                i = i + 2
            else
                in_quotes = false
                i = i + 1
            end
        else
            if #field == 0 or self.lazy_quotes then
                in_quotes = true
                i = i + 1
            else
                table.insert(field, c)
                i = i + 1
            end
        end
        elseif c == sep and not in_quotes then
            push_field()
            i = i + 1
        else
            table.insert(field, c)
            i = i + 1
        end
    end

    push_field()
    return out
end

function CSV:_cast_fields(fields)
    if not self.infer and not self.na then return fields end
    local na = self.na and list_to_set(self.na, false) or nil
    for i = 1, #fields do
        local v = fields[i]
        if na and na[v] then
            fields[i] = nil
        elseif self.infer then
            local num = tonumber(v)
            if num ~= nil then fields[i] = num end
        end
    end
    return fields
end

function CSV:set_excluded(names, opts)
    assert(type(names) == 'table', 'set_excluded expects a list of names')
    opts = opts or {}
    self._exclude_ci = opts.case_insensitive and true or false
    self._exclude = list_to_set(names, self._exclude_ci)
    self._exclude_col = opts.col or opts.key
    return self
end

function CSV:clear_excluded()
    self._exclude = nil
    self._exclude_col = nil
    self._exclude_ci = false
    return self
end

function CSV:set_excluded_columns(names, opts)
    assert(type(names) == 'table', 'set_excluded_columns expects a list of header names')
    self._drop_cols_ci = (opts and opts.case_insensitive) and true or false
    self._drop_cols = list_to_set(names, self._drop_cols_ci)
    self._drop_cols_idx = nil
    return self
end

function CSV:clear_excluded_columns()
    self._drop_cols = nil
    self._drop_cols_ci = false
    self._drop_cols_idx = nil
    return self
end

function CSV:fit_normalizer(source, columns, opts)
    opts = opts or {}
    local method = (opts.method or 'zscore'):lower()
    assert(method == 'zscore' or method == 'minmax', 'method must be zscore or minmax')
    local eps = opts.eps or 1e-12
    local ci = opts.case_insensitive and true or false

    local iter, state = self:rows(source)
    local first = iter()
    if not first then
        if state and state.close then pcall(state.close) end
        self._norm = { method = method, eps = eps, ci = ci, by_name = {}, by_index = {} }
        return self
    end

    local use_names = self.header and type(first) == 'table' and (first[1] == nil)
    local by_name, by_index = {}, {}

    if columns == nil then
        if use_names then
            for k, v in pairs(first) do
                if type(v) == 'number' then by_name[k] = {n=0, mean=0, m2=0, min=nil, max=nil} end
            end
        else
            for j = 1, #first do
                if type(first[j]) == 'number' then by_index[j] = {n=0, mean=0, m2=0, min=nil, max=nil} end
            end
        end
    else
        if use_names then
            local lookup = {}
            for k, _ in pairs(first) do lookup[ci and tostring(k):lower() or k] = k end
            for _, name in ipairs(columns) do
                local key = ci and tostring(name):lower() or name
                local real = lookup[key]
                if real then by_name[real] = {n=0, mean=0, m2=0, min=nil, max=nil} end
            end
        else
            for _, j in ipairs(columns) do
                if type(j) == 'number' and j >= 1 then by_index[j] = {n=0, mean=0, m2=0, min=nil, max=nil} end
            end
        end
    end

    local function update(stat, x)
        if type(x) ~= 'number' then return end
        stat.n = stat.n + 1
        local delta = x - stat.mean
        stat.mean = stat.mean + delta / stat.n
        local delta2 = x - stat.mean
        stat.m2 = stat.m2 + delta * delta2
        if stat.min == nil or x < stat.min then stat.min = x end
        if stat.max == nil or x > stat.max then stat.max = x end
    end

    if use_names then
        for k, stat in pairs(by_name) do update(stat, first[k]) end
    else
        for j, stat in pairs(by_index) do update(stat, first[j]) end
    end
    for row in iter do
        if use_names then
            for k, stat in pairs(by_name) do update(stat, row[k]) end
        else
            for j, stat in pairs(by_index) do update(stat, row[j]) end
        end
    end
    if state and state.close then pcall(state.close) end

    for _, stat in pairs(by_name) do
        stat.std = (stat.n > 1) and math.sqrt(stat.m2 / (stat.n - 1)) or 0
    end
    for _, stat in pairs(by_index) do
        stat.std = (stat.n > 1) and math.sqrt(stat.m2 / (stat.n - 1)) or 0
    end

    self._norm = { method = method, eps = eps, ci = ci, by_name = by_name, by_index = by_index }
    return self
end

function CSV:clear_normalizer()
    self._norm = nil
    return self
end

------------------------------------------------------------
-- Public
------------------------------------------------------------
function CSV:rows(source)
    local handle, close_when_done = nil, false

    if type(source) == 'string' then
        local f = io.open(source, 'r')
        if f then
            handle = f
            close_when_done = true
        else
            handle = make_string_reader(source)
        end
    elseif type(source) == 'userdata' or type(source) == 'table' then
        handle = source
    else
        error('Unsupported source: expected filename, file handle, or CSV string')
    end

    local header = nil
    local done = false

    local function closer()
        if close_when_done and handle and handle.close then pcall(function() handle:close() end) end
    end

    local function is_excluded_row(row, fields)
        if not self._exclude then return false end
        local key = self._exclude_col
        if key == nil then
            key = header and 'name' or 1
            self._exclude_col = key
        end

        local val
        if header then
            if type(key) == 'number' then
                val = fields and fields[key] or nil
            else
                val = row[key]
            end
        else
            if type(key) == 'number' then
                val = row[key]
            else
                return false
            end
        end

        if val == nil then return false end
        if self._exclude_ci then
            return self._exclude[tostring(val):lower()] == true
        else
            return self._exclude[val] == true
        end
    end

    local function compute_drop_indices()
        if not (self._drop_cols and header and not self._drop_cols_idx) then return end
        self._drop_cols_idx = {}
        for i = 1, #header do
            local name = header[i]
            local key = self._drop_cols_ci and tostring(name):lower() or name
            if self._drop_cols[key] then
                self._drop_cols_idx[i] = true
            end
        end
    end

    local function apply_normalization(row)
        if not self._norm then return end
        local N = self._norm
        if header and N.by_name then
            for k, stat in pairs(N.by_name) do
                local v = row[k]
                if type(v) == 'number' then
                    if N.method == 'zscore' then
                        local denom = (stat.std and stat.std > 0) and stat.std or N.eps
                        row[k] = (v - (stat.mean or 0)) / denom
                    else
                        local range = ((stat.max or 0) - (stat.min or 0))
                        if range <= 0 then range = N.eps end
                        row[k] = (v - (stat.min or 0)) / range
                    end
                end
            end
        elseif not header and N.by_index then
            for j, stat in pairs(N.by_index) do
                local v = row[j]
                if type(v) == 'number' then
                    if N.method == 'zscore' then
                        local denom = (stat.std and stat.std > 0) and stat.std or N.eps
                        row[j] = (v - (stat.mean or 0)) / denom
                    else
                        local range = ((stat.max or 0) - (stat.min or 0))
                        if range <= 0 then range = N.eps end
                        row[j] = (v - (stat.min or 0)) / range
                    end
                end
            end
        end
    end

    local function next_row()
        if done then return nil end
        local rec = self:_read_record(handle)
        if not rec then
            done = true
            closer()
            return nil
        end

        if rec == '' then
            return next_row()
        end

        local fields = self:_parse_record(rec)
        fields = self:_cast_fields(fields)

        if self.header and not header then
            header = fields
            compute_drop_indices()
            return next_row()
        end

        if header then
            compute_drop_indices()
            local row = {}
            for i = 1, #header do
                if not (self._drop_cols_idx and self._drop_cols_idx[i]) then
                    row[header[i]] = fields[i]
                end
            end
            if is_excluded_row(row, fields) then
                return next_row()
            end
            apply_normalization(row)
            return row
        else
            if is_excluded_row(fields) then
                return next_row()
            end
            apply_normalization(fields)
            return fields
        end
    end

    local iter_state = { close = closer }
    return function()
        return next_row()
    end, iter_state
end

function CSV.read_all(source, opts)
    local reader = CSV.new(opts)
    local iter, state = reader:rows(source)
    local rows = {}
    for row in iter do table.insert(rows, row) end
    if state and state.close then pcall(state.close) end
    return rows
end

function CSV:min_max(source)
    local iter, state = self:rows(source)

    local mins, maxs = {}, {}

    for row in iter do
        if self.header then
            for k, v in pairs(row) do
                if type(v) == 'number' then
                    local cur_min = mins[k]
                    local cur_max = maxs[k]
                    if cur_min == nil or v < cur_min then mins[k] = v end
                    if cur_max == nil or v > cur_max then maxs[k] = v end
                end
            end
        else
            for i = 1, #row do
                local v = row[i]
                if type(v) == 'number' then
                    local cur_min = mins[i]
                    local cur_max = maxs[i]
                    if cur_min == nil or v < cur_min then mins[i] = v end
                    if cur_max == nil or v > cur_max then maxs[i] = v end
                end
            end
        end
    end

    if state and state.close then pcall(state.close) end
    return mins, maxs
end


function CSV:count_rows(source)
    local iter, state = self:rows(source)
    local n = 0
    for _ in iter do n = n + 1 end
    if state and state.close then pcall(state.close) end
    return n
end

function CSV:headers(source)
    if not self.header then return nil end
    local handle, close_when_done = nil, false
    if type(source) == 'string' then
        local f = io.open(source, 'r')
        if f then handle = f; close_when_done = true else handle = make_string_reader(source) end
    elseif type(source) == 'userdata' or type(source) == 'table' then
        handle = source
    else
        error('Unsupported source: expected filename, file handle, or CSV string')
    end
    local rec = self:_read_record(handle)
    local hdr = rec and self:_parse_record(rec) or nil
    if close_when_done and handle and handle.close then pcall(function() handle:close() end) end
    return hdr
end

function CSV:peek(source, n)
    n = n or 5
    local iter, state = self:rows(source)
    local out, k = {}, 0
    for row in iter do
        k = k + 1
        out[k] = row
        if k >= n then break end
    end
    if state and state.close then pcall(state.close) end
    return out
end

function CSV:shape(source)
    local nrows = self:count_rows(source)
    local iter, state = self:rows(source)
    local first = iter()
    if state and state.close then pcall(state.close) end
    local ncols = 0
    if first then
        if self.header and type(first) == 'table' then
            for _ in pairs(first) do ncols = ncols + 1 end
        else
            ncols = #first
        end
    end
    return nrows, ncols
end

------------------------------------------------------------
-- Module-level wrappers
------------------------------------------------------------
local M = {}
M.new        = CSV.new
M.rows       = function(source, opts) return CSV.new(opts):rows(source) end
M.read_all   = function(source, opts) return CSV.read_all(source, opts) end
M.count_rows = function(source, opts) return CSV.new(opts):count_rows(source) end
M.headers    = function(source, opts) return CSV.new(opts):headers(source) end
M.peek       = function(source, n, opts) return CSV.new(opts):peek(source, n) end
M.shape      = function(source, opts) return CSV.new(opts):shape(source) end

return M

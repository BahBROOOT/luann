local sep = package.config:sub(1, 1)

local function file_exists(path)
    local f = io.open(path, "r")
    if f then
        f:close()
        return true
    end
    return false
end

local function dirname(path)
    return path:match("^(.*[/\\])") or "." .. sep
end

local source = debug.getinfo(1, "S").source or ""
if source:sub(1, 1) == "@" then
    source = source:sub(2)
end

local here = dirname(source)
local root = here

if file_exists(here .. "nn.lua") then
    root = here
elseif file_exists(here .. ".." .. sep .. "nn.lua") then
    root = here .. ".." .. sep
elseif file_exists("." .. sep .. "nn.lua") then
    root = "." .. sep
elseif file_exists(".." .. sep .. "nn.lua") then
    root = ".." .. sep
end

local paths = {
    root .. "?.lua",
    root .. "?" .. sep .. "init.lua",
    root .. "tests" .. sep .. "?.lua",
}

for i=#paths,1,-1 do
    local p = paths[i]
    if not package.path:find(p, 1, true) then
        package.path = p .. ";" .. package.path
    end
end

return { root = root }

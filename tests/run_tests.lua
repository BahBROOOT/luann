local ok, err = pcall(require, "tests.bootstrap")
if not ok then
    local ok_local, err_local = pcall(require, "bootstrap")
    if not ok_local then
        error(tostring(err) .. "\n" .. tostring(err_local), 2)
    end
end

local tests = {
    "tests.cpu_backend_test",
    "tests.gpu_backend_test",
}

for _,name in ipairs(tests) do
    package.loaded[name] = nil
    require(name)
end

print("run_tests: ok")

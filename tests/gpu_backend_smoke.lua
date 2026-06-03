local ok, err = pcall(require, "tests.bootstrap")
if not ok then
    local ok_local, err_local = pcall(require, "bootstrap")
    if not ok_local then
        error(tostring(err) .. "\n" .. tostring(err_local), 2)
    end
end

require("tests.gpu_backend_test")
print("gpu_backend_smoke: ok")

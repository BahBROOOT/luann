local ok, err = pcall(require, "tests.bootstrap")
if not ok then
    local ok_local, err_local = pcall(require, "bootstrap")
    if not ok_local then
        error(tostring(err) .. "\n" .. tostring(err_local), 2)
    end
end

local NN = require("nn")
local T = require("tests.test_helper")

local X = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1},
}

local y = {1, 2, 2, 1}

local devices = NN.available_devices()
assert(type(devices) == "table" and #devices >= 1, "available_devices should return at least CPU")
assert(devices[1].device == "cpu" and devices[1].available == true, "CPU device should always be available")
assert(type(NN.gpu_info()) == "table", "gpu_info should return a table")

NN.RNG:randomseed(123)
local cpu = T.build_classifier(NN)
local gpu = T.build_classifier(NN)
NN.load_state(gpu, NN.save_state(cpu))

local cpu_logits = cpu:forward(X, { device = "cpu" })
local gpu_or_fallback_logits = gpu:forward(X, { device = "gpu" })

if NN.gpu_available() then
    T.matrix_close(cpu_logits, gpu_or_fallback_logits, 1e-4, "gpu forward")

    local cpu_probs = cpu:predict_proba(X, { device = "cpu" })
    local gpu_probs = gpu:predict_proba(X, { device = "gpu" })
    T.matrix_close(cpu_probs, gpu_probs, 1e-4, "gpu predict_proba")
    T.prob_rows(gpu_probs, 1e-5, "gpu probabilities")

    local auto = T.build_classifier(NN)
    NN.load_state(auto, NN.save_state(cpu))
    local auto_probs = auto:predict_proba(X, { device = "auto" })
    T.matrix_close(gpu_probs, auto_probs, 1e-4, "gpu auto predict_proba")

    NN.RNG:randomseed(456)
    cpu:fit(X, y, { epochs = 2, batch = 2, lr = 0.01, algo = "adam", verbose = false, device = "cpu" })
    NN.RNG:randomseed(456)
    gpu:fit(X, y, { epochs = 2, batch = 2, lr = 0.01, algo = "adam", verbose = false, device = "gpu" })

    local cpu_after = cpu:predict_proba(X, { device = "cpu" })
    local gpu_after = gpu:predict_proba(X, { device = "gpu" })
    T.matrix_close(cpu_after, gpu_after, 2e-3, "gpu fit parity")

    local reloaded = T.build_classifier(NN)
    NN.load_state(reloaded, NN.save_state(gpu))
    T.matrix_close(
        gpu_after,
        reloaded:predict_proba(X, { device = "cpu" }),
        1e-6,
        "gpu serialization"
    )
else
    T.matrix_close(cpu_logits, gpu_or_fallback_logits, 1e-9, "gpu fallback forward")

    local ok, err = pcall(function()
        gpu:forward(X, { device = "gpu", gpu_required = true })
    end)
    assert(not ok, "gpu_required should fail when GPU is unavailable")
    assert(tostring(err):match("GPU backend requested but unavailable"), "gpu_required error should explain GPU unavailability")
    if not jit then
        assert(
            tostring(err):match("cannot execute GPU tensor compute yet") and tostring(err):match("LuaJIT"),
            "plain Lua GPU error should explain that GPU tensor compute currently requires LuaJIT"
        )
    end

    local auto = T.build_classifier(NN)
    local auto_logits = auto:forward(X, { device = "auto" })
    assert(type(auto_logits) == "table", "auto should fall back to CPU when GPU is unavailable")
end

print("gpu_backend_test: ok")

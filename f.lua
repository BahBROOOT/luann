------------------------------------------------------------
-- Heavy / stressful compute example
--
-- This is intentionally much larger than example.lua. It is meant to make
-- the GPU do real work. Presets are intentionally tiered:
--
--   small   = quick sanity check
--   medium  = visible GPU work
--   stress  = I paid for the whole GPU
--   brutal  = go get coffee
--   absurd  = I accept the consequences
--
-- Run from the repo root:
--
--   luajit f.lua gpu medium
--   luajit f.lua gpu stress
--   luajit f.lua gpu brutal
--   luajit f.lua gpu absurd
--   lua f.lua cpu small
--
--   luajit f.lua gpu 32768 10 512
--
-- Args:
--   1: device  = cpu | gpu | auto
--   2: preset  = small | medium | stress | brutal | absurd
--      or samples as a number for legacy/custom mode
--   3+: legacy/custom numbers:
--      samples epochs hidden features classes depth batch
------------------------------------------------------------

local NN = require("nn")
local M = NN.M
local RNG = NN.RNG

local unpack_values = table.unpack or unpack

local device = (arg and arg[1]) or "auto"
device = tostring(device):lower()

local presets = {
    small = {
        samples = 4096,
        epochs = 4,
        hidden = 128,
        features = 64,
        classes = 6,
        depth = 4,
        batch = 256,
    },
    medium = {
        samples = 8192,
        epochs = 5,
        hidden = 256,
        features = 96,
        classes = 8,
        depth = 5,
        batch = 256,
    },
    stress = {
        samples = 16384,
        epochs = 6,
        hidden = 384,
        features = 128,
        classes = 8,
        depth = 5,
        batch = 384,
    },
    brutal = {
        samples = 32768,
        epochs = 8,
        hidden = 512,
        features = 192,
        classes = 10,
        depth = 6,
        batch = 512,
    },
    absurd = {
        samples = 65536,
        epochs = 12,
        hidden = 768,
        features = 256,
        classes = 12,
        depth = 8,
        batch = 768,
    },
}

local function clone(t)
    local out = {}
    for k,v in pairs(t) do out[k] = v end
    return out
end

local default_preset = (jit and device ~= "cpu") and "stress" or "small"
local preset_name = (arg and arg[2]) or default_preset
local cfg

if tonumber(preset_name) then
    cfg = clone(presets.stress)
    cfg.samples = tonumber(arg[2]) or cfg.samples
    cfg.epochs = tonumber(arg[3]) or cfg.epochs
    cfg.hidden = tonumber(arg[4]) or cfg.hidden
    cfg.features = tonumber(arg[5]) or cfg.features
    cfg.classes = tonumber(arg[6]) or cfg.classes
    cfg.depth = tonumber(arg[7]) or cfg.depth
    cfg.batch = tonumber(arg[8]) or cfg.batch
    preset_name = "custom"
else
    preset_name = tostring(preset_name):lower()
    cfg = clone(presets[preset_name] or presets[default_preset])
end

local function seconds(label, fn)
    local t0 = os.clock()
    local results = { fn() }
    local dt = os.clock() - t0
    print(("%-20s %.3fs"):format(label .. ":", dt))
    return unpack_values(results)
end

local function approx_mb()
    local x = cfg.samples * cfg.features * 4
    local y = cfg.samples * cfg.classes * 4
    local weights = cfg.features * cfg.hidden * 4
    weights = weights + math.max(0, cfg.depth - 1) * cfg.hidden * cfg.hidden * 4
    weights = weights + cfg.hidden * cfg.classes * 4
    return (x + y + weights * 8) / (1024 * 1024)
end

local function matmul_ops_per_epoch()
    local first = cfg.samples * cfg.features * cfg.hidden
    local middle = cfg.samples * cfg.hidden * cfg.hidden * math.max(0, cfg.depth - 1)
    local last = cfg.samples * cfg.hidden * cfg.classes
    -- Rough forward + backward multiplier for dense layers.
    return 3 * 2 * (first + middle + last)
end

local function print_gpu_summary()
    if not NN.gpu_info then return end
    local info = NN.gpu_info()
    print("GPU available:       ", NN.gpu_available())
    if info.backend then print("GPU backend:         ", info.backend) end
    if info.device then print("GPU device:          ", info.device) end
    if info.reason then print("GPU note:            ", info.reason) end
    if info.native_probe and info.native_probe.device then
        print("Bridge probe:        ", info.native_probe.device)
    end
end

local function make_heavy_data(n, d, c)
    local X = M.zeros(n, d)
    local y = {}

    for i=1,n do
        local row = X[i]
        local phase = i * 0.0097
        for j=1,d do
            local a = math.sin(phase + j * 0.113)
            local b = math.cos(i * j * 0.00019 + j * 0.071)
            local mix = math.sin((i % 257) * (j % 31) * 0.003)
            local noise = (RNG:rand() - 0.5) * 0.05
            row[j] = a + 0.7 * b + 0.35 * mix + noise
        end

        local best_score = -1e99
        local best_class = 1
        for k=1,c do
            local s = 0.0
            for j=1,d do
                local w1 = math.sin(k * 0.41 + j * 0.071)
                local w2 = math.cos(k * j * 0.017)
                s = s + row[j] * (w1 + 0.65 * w2)
            end
            s = s + 0.8 * math.sin(row[1] * row[2] + k * 0.33)
            s = s + 0.4 * math.cos(row[3] * row[4] - k * 0.27)
            if s > best_score then
                best_score = s
                best_class = k
            end
        end

        y[i] = best_class
    end

    return X, y
end

local function build_net()
    local net = NN.new({ device = device })
    net:add(NN.Linear(cfg.features, cfg.hidden))

    for i=1,cfg.depth-1 do
        if i % 2 == 1 then
            net:add(NN.ReLU())
        else
            net:add(NN.Tanh())
        end
        net:add(NN.Linear(cfg.hidden, cfg.hidden))
    end

    net:add(NN.ReLU())
    net:add(NN.Linear(cfg.hidden, cfg.classes))
    return net
end

local function accuracy(pred, y)
    local correct = 0
    for i=1,#y do
        if pred[i] == y[i] then correct = correct + 1 end
    end
    return correct / #y
end

local function fmt_big(n)
    if n >= 1e12 then return ("%.2fT"):format(n / 1e12) end
    if n >= 1e9 then return ("%.2fB"):format(n / 1e9) end
    if n >= 1e6 then return ("%.2fM"):format(n / 1e6) end
    return tostring(math.floor(n))
end

print("== luann GPU stress example ==")
print(("runtime:             %s"):format(jit and "LuaJIT" or _VERSION))
print(("device:              %s"):format(device))
print(("preset:              %s"):format(preset_name))
print(("samples:             %d"):format(cfg.samples))
print(("features:            %d"):format(cfg.features))
print(("classes:             %d"):format(cfg.classes))
print(("hidden:              %d"):format(cfg.hidden))
print(("hidden layers:       %d"):format(cfg.depth))
print(("epochs:              %d"):format(cfg.epochs))
print(("batch:               %d"):format(cfg.batch))
print(("rough dense ops/ep:  %s"):format(fmt_big(matmul_ops_per_epoch())))
print(("rough GPU memory:    %.1f MB"):format(approx_mb()))

if not jit and device ~= "cpu" then
    print("plain Lua note:      GPU tensor compute requires LuaJIT today; this will fall back to CPU.")
end
if preset_name == "brutal" then
    print("stress note:         Brutal is a coffee-break workload.")
elseif preset_name == "absurd" then
    print("stress note:         Absurd is the danger zone; expect high VRAM use and long runtimes.")
end

print_gpu_summary()

RNG:randomseed(2026)

local X, y = seconds("data generation", function()
    return make_heavy_data(cfg.samples, cfg.features, cfg.classes)
end)

local net = build_net()

seconds("initial forward", function()
    return net:forward(X, { device = device })
end)

seconds("training", function()
    net:fit(X, y, {
        epochs = cfg.epochs,
        batch = cfg.batch,
        lr = 0.001,
        algo = "adam",
        verbose = false,
        device = device,
        num_classes = cfg.classes,
    })
end)

local preds = seconds("prediction", function()
    return net:predict_classes(X, { device = device })
end)

print(("train accuracy:      %.2f%%"):format(100 * accuracy(preds, y)))

local probe = {
    X[1],
    X[math.max(1, math.floor(cfg.samples / 3))],
    X[math.max(1, math.floor(cfg.samples * 2 / 3))],
    X[cfg.samples],
}

local probs = seconds("probe proba", function()
    return net:predict_proba(probe, { device = device })
end)

print("probe probabilities:")
for i=1,#probs do
    local parts = {}
    for j=1,#probs[i] do
        parts[j] = ("%.3f"):format(probs[i][j])
    end
    print(("  %d: %s"):format(i, table.concat(parts, "  ")))
end

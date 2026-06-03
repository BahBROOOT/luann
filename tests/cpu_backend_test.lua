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
    {1, 2},
    {-1, 0.5},
}

local net = T.build_linear_net(NN)
local logits = net:forward(X, { device = "cpu" })
T.matrix_close(logits, {
    {7.5, 9.5},
    {1.0, -0.5},
}, 1e-9, "cpu forward")

local probs = net:predict_proba(X, { device = "cpu" })
T.prob_rows(probs, 1e-6, "cpu predict_proba")
T.vector_equal(net:predict_classes(X, { device = "cpu" }), {2, 1}, "cpu predict_classes")

local cpu_default = NN.new({ device = "cpu" })
cpu_default:add(NN.Linear(2, 2))
NN.load_state(cpu_default, NN.save_state(net))
T.matrix_close(cpu_default:forward(X), logits, 1e-9, "cpu default device")

local train_X = {
    {-2, -1},
    {-1, -2},
    {1, 2},
    {2, 1},
}
local train_y = {1, 1, 2, 2}

NN.RNG:randomseed(99)
local train_net = T.build_classifier(NN)
local before = T.snapshot_matrix(train_net.layers[1].W)
train_net:fit(train_X, train_y, {
    epochs = 8,
    batch = 2,
    lr = 0.02,
    algo = "adam",
    verbose = false,
    device = "cpu",
})

local after = train_net.layers[1].W
local changed = false
for i=1,#before do
    for j=1,#before[i] do
        if math.abs(before[i][j] - after[i][j]) > 1e-9 then
            changed = true
        end
    end
end
assert(changed, "cpu fit should update weights")

local state = NN.save_state(train_net)
local loaded = T.build_classifier(NN)
NN.load_state(loaded, state)
T.matrix_close(
    loaded:predict_proba(train_X, { device = "cpu" }),
    train_net:predict_proba(train_X, { device = "cpu" }),
    1e-9,
    "cpu serialization"
)

print("cpu_backend_test: ok")

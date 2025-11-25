------------------------------------------------------------
-- Init
------------------------------------------------------------

local NN = require('nn_cpu')
local M = NN.M
local RNG = NN.RNG
local csv = require('csv')

------------------------------------------------------------
-- Make data
------------------------------------------------------------
local function make_circle_data(N)
    local X = M.zeros(N, 2)
    local y = {}
    local r = 1.2
    local r2 = r * r
    for i=1,N do
        local x = -2 + 4*RNG:rand()
        local z = -2 + 4*RNG:rand()
        X[i][1], X[i][2] = x, z
        local inside = (x*x + z*z) < r2
        y[i] = inside and 1 or 2   -- labels must be 1..C (Lua is 1-based)
    end
    return X, y
end

local X, y = make_circle_data(800)

------------------------------------------------------------
-- Build, Train, Test net
------------------------------------------------------------
local net = NN.new()
net:add(NN.Linear(2, 32))
net:add(NN.Tanh())
net:add(NN.Linear(32, 32))
net:add(NN.Tanh())
net:add(NN.Linear(32, 2))   -- 2 classes: inside vs outside

-- TRAIN
net:fit(X, y, {epochs=200, batch=32, lr=0.002, algo="adam", verbose=true})

-- EVAL
local preds = net:predict_classes(X)
local acc = NN.metrics.accuracy_from_indices(preds, y)
print(("Train accuracy: %.1f%%"):format(100*acc))

-- TEST
local test = { {0,0}, {1.1,0}, {1.5,0}, {0.9,0.9}, {1.3,0.3} }
local probs = net:predict_proba(test)
print("\nProbs (class1=inside, class2=outside):")
for i=1,#test do
    local x,z = test[i][1], test[i][2]
    print(("(%.1f,%.1f) -> %.3f  %.3f"):format(x, z, probs[i][1], probs[i][2]))
end



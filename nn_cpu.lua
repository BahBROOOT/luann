------------------------------------------------------------
-- Init
-- See doc/nn-usage.md for instructions.
-- Example: luajit nn_cpu.lua demo
-- Example: lua54 nn_cpu.lua demo
------------------------------------------------------------

--[=[
local version = 'Lua 5.0'
--[[]=]
local n = '8'; repeat n = n*n until n == n*n
local t = {'Lua 5.1', nil,
  [-1/0] = 'Lua 5.2',
  [1/0]  = 'Lua 5.3',
  [2]    = 'LuaJIT'}
local version = t[2] or t[#'\z'] or t[n/'-0'] or 'Lua 5.4'
--]]
local case = {
    ["Lua 5.0"] = ("nn: Old Lua version detected (%s) expect problems. Recommended (Lua 5.3+ or LuaJIT)"):format(version),
    ["Lua 5.1"] = ("nn: Old Lua version detected (%s) expect problems. Recommended (Lua 5.3+ or LuaJIT)"):format(version),
    ["Lua 5.2"] = ("nn: Old Lua version detected (%s) expect problems. Recommended (Lua 5.3+ or LuaJIT)"):format(version),
    ["Lua 5.3"] = ("nn: Current Lua version (%s). This version will work fine for this script but is slower than the recommended LuaJIT version, wich will run faster!"):format(version),
    ["Lua 5.4"] = ("nn: Current Lua version (%s). This version will work fine for this script but is slower than the recommended LuaJIT version, wich will run faster!"):format(version),
    ["LuaJIT"] = ("nn: Current Lua version (%s)."):format(version),
}
print(case[version])

local NN = {}

-- NN.backend = {}
-- NN.isLuaJIT = version == "LuaJIT" and true or false

------------------------------------------------------------
-- Utility
------------------------------------------------------------
local function printf(fmt, ...)
    io.write((string.format(fmt, ...)) .. "\n")
end

local function idiv(a,b) return math.floor(a/b) end
local function imod(a,b) return a - b*math.floor(a/b) end

local function fast_tanh(x)
    if x > 20 then return 1.0 elseif x < -20 then return -1.0 end
    local t = math.exp(-2*x)
    return (1 - t) / (1 + t)
end

------------------------------------------------------------
-- RNG helpers
------------------------------------------------------------
local RNG = { seed = os.time() % 2147483647 }
function RNG:randomseed(s)
    self.seed = imod((s or os.time()), 2147483647)
end

function RNG:rand()
    self.seed = imod(1103515245 * self.seed + 12345, 2147483647)
    return self.seed / 2147483647
end

-- Box-Muller for ~N(0,1)
function RNG:randn()
    local u1 = 1 - self:rand()
    local u2 = 1 - self:rand()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
end

------------------------------------------------------------
-- Matrix helpers
------------------------------------------------------------
-- local M = NN.backend
local M = {}

function M.shape(A)
    return #A, (#A > 0 and #A[1] or 0)
end

function M.zeros(n, m)
    local A = {}
    for i=1,n do
        local row = {}
        for j=1,m do row[j] = 0.0 end
        A[i] = row
    end
    return A
end

function M.ones(n, m)
    local A = {}
    for i=1,n do
        local row = {}
        for j=1,m do row[j] = 1.0 end
        A[i] = row
    end
    return A
end

function M.randn(n, m, scale)
    scale = scale or 1.0
    local A = {}
    for i=1,n do
        local row = {}
        for j=1,m do row[j] = RNG:randn() * scale end
        A[i] = row
    end
    return A
end

function M.copy(A)
    local n,m = M.shape(A)
    local B = {}
    for i=1,n do
        local row = {}
        for j=1,m do row[j] = A[i][j] end
        B[i] = row
    end
    return B
end

function M.transpose(A)
    local n,m = M.shape(A)
    local T = M.zeros(m,n)
    for i=1,n do
        for j=1,m do
            T[j][i] = A[i][j]
        end
    end
    return T
end

function M.matmul(A, B)
    local n, mA = M.shape(A)
    local mB, p = M.shape(B)
    assert(mA == mB, "matmul dim mismatch")
    local C = M.zeros(n, p)
    for i=1,n do
        for k=1,p do
            local s = 0.0
            local Ai = A[i]
            for j=1,mA do
                s = s + Ai[j] * B[j][k]
            end
            C[i][k] = s
        end
    end
    return C
end

function M.add(A, B)
    local n,m = M.shape(A)
    local nB,mB = M.shape(B)
    local C = M.zeros(n,m)
    if nB == 1 and mB == m then
        local b = B[1]
        for i=1,n do
            local Ai = A[i]; local Ci = C[i]
            for j=1,m do Ci[j] = Ai[j] + b[j] end
        end
    else
        assert(n==nB and m==mB, "add shape mismatch")
        for i=1,n do
            local Ai = A[i]; local Bi = B[i]; local Ci = C[i]
            for j=1,m do Ci[j] = Ai[j] + Bi[j] end
        end
    end
    return C
end

function M.sub(A, B)
    local n,m = M.shape(A)
    local nB,mB = M.shape(B)
    local C = M.zeros(n,m)
    if nB == 1 and mB == m then
        local b = B[1]
        for i=1,n do
            local Ai = A[i]; local Ci = C[i]
            for j=1,m do Ci[j] = Ai[j] - b[j] end
        end
    else
        assert(n==nB and m==mB, "sub shape mismatch")
        for i=1,n do
            local Ai = A[i]; local Bi = B[i]; local Ci = C[i]
            for j=1,m do Ci[j] = Ai[j] - Bi[j] end
        end
    end
    return C
end

function M.hadamard(A, B)
    local n,m = M.shape(A)
    local nB,mB = M.shape(B)
    assert(n==nB and m==mB, "hadamard shape mismatch")
    local C = M.zeros(n,m)
    for i=1,n do
        local Ai = A[i]; local Bi = B[i]; local Ci = C[i]
        for j=1,m do Ci[j] = Ai[j] * Bi[j] end
    end
    return C
end

function M.scalar_add(A, s)
    local n,m = M.shape(A)
    local C = M.zeros(n,m)
    for i=1,n do
        local Ai = A[i]; local Ci = C[i]
        for j=1,m do Ci[j] = Ai[j] + s end
    end
    return C
end

function M.scalar_mul(A, s)
    local n,m = M.shape(A)
    local C = M.zeros(n,m)
    for i=1,n do
        local Ai = A[i]; local Ci = C[i]
        for j=1,m do Ci[j] = Ai[j] * s end
    end
    return C
end

function M.sum(A, axis)
    local n,m = M.shape(A)
    if axis == 1 then
        local r = M.zeros(1, m)
        local rr = r[1]
        for i=1,n do
            local Ai = A[i]
            for j=1,m do rr[j] = rr[j] + Ai[j] end
        end
        return r
    elseif axis == 2 then
        local r = M.zeros(n, 1)
        for i=1,n do
            local s = 0.0
            local Ai = A[i]
            for j=1,m do s = s + Ai[j] end
            r[i][1] = s
        end
        return r
    else
        local s = 0.0
        for i=1,n do for j=1,m do s = s + A[i][j] end end
        return s
    end
end

function M.apply(A, fn)
    local n,m = M.shape(A)
    local B = M.zeros(n,m)
    for i=1,n do
        local Ai = A[i]; local Bi = B[i]
        for j=1,m do Bi[j] = fn(Ai[j]) end
    end
    return B
end

function M.argmax_rowwise(A)
    local n,m = M.shape(A)
    local idx = {}
    for i=1,n do
        local Ai = A[i]
        local maxv = Ai[1]
        local maxj = 1
        for j=2,m do
            if Ai[j] > maxv then maxv = Ai[j]; maxj = j end
        end
        idx[i] = maxj
    end
    return idx
end

function M.one_hot(y, num_classes)
    local n = #y
    local Y = M.zeros(n, num_classes)
    for i=1,n do Y[i][y[i]] = 1.0 end
    return Y
end

------------------------------------------------------------
-- Layers
------------------------------------------------------------
local Layer = {}
Layer.__index = Layer

local Linear = setmetatable({}, Layer)
Linear.__index = Linear

function Linear:new(in_dim, out_dim, opts)
    opts = opts or {}
    local scale = (opts.init_scale or math.sqrt(2/(in_dim)))
    local W = M.randn(in_dim, out_dim, scale)
    local b = M.zeros(1, out_dim)
    local o = {
        type = "Linear",
        in_dim = in_dim,
        out_dim = out_dim,
        W = W, b = b,
        dW = M.zeros(in_dim, out_dim),
        db = M.zeros(1, out_dim),
        mW = M.zeros(in_dim, out_dim), vW = M.zeros(in_dim, out_dim),
        mB = M.zeros(1, out_dim),     vB = M.zeros(1, out_dim),
        t = 0,
    }
    return setmetatable(o, Linear)
end

function Linear:forward(x)
    self.x = x
    self.out = M.add(M.matmul(x, self.W), self.b)
    return self.out
end

function Linear:backward(dout)
    local xT = M.transpose(self.x)
    self.dW = M.matmul(xT, dout)
    self.db = M.sum(dout, 1)
    local WT = M.transpose(self.W)
    local dx = M.matmul(dout, WT)
    return dx
end

local Activation = setmetatable({}, Layer)
Activation.__index = Activation

local ReLU = setmetatable({}, Activation)
ReLU.__index = ReLU
function ReLU:new() return setmetatable({ type="ReLU" }, ReLU) end
function ReLU:forward(x)
    self.out = M.apply(x, function(v) return v > 0 and v or 0 end)
    self.mask = M.apply(x, function(v) return v > 0 and 1.0 or 0.0 end)
    return self.out
end
function ReLU:backward(dout)
    return M.hadamard(dout, self.mask)
end

local Sigmoid = setmetatable({}, Activation)
Sigmoid.__index = Sigmoid
function Sigmoid:new() return setmetatable({ type="Sigmoid" }, Sigmoid) end
function Sigmoid:forward(x)
    self.out = M.apply(x, function(v) return 1.0/(1.0+math.exp(-v)) end)
    return self.out
end
function Sigmoid:backward(dout)
    local y = self.out
    local one_minus_y = M.scalar_add(M.scalar_mul(y, -1), 1.0)
    local dydx = M.hadamard(y, one_minus_y)
    return M.hadamard(dout, dydx)
end

local Tanh = setmetatable({}, Activation)
Tanh.__index = Tanh
function Tanh:new() return setmetatable({ type="Tanh" }, Tanh) end
function Tanh:forward(x)
    self.out = M.apply(x, function(v) return fast_tanh(v) end)
    return self.out
end
function Tanh:backward(dout)
    local y = self.out
    local y2 = M.hadamard(y, y)
    local dydx = M.scalar_add(M.scalar_mul(y2, -1.0), 1.0)
    return M.hadamard(dout, dydx)
end

------------------------------------------------------------
-- Losses
------------------------------------------------------------
local Loss = {}
Loss.__index = Loss

local SoftmaxCrossEntropy = setmetatable({}, Loss)
SoftmaxCrossEntropy.__index = SoftmaxCrossEntropy
function SoftmaxCrossEntropy:new()
    return setmetatable({ type="SoftmaxCrossEntropy" }, SoftmaxCrossEntropy)
end
function SoftmaxCrossEntropy:forward(logits, target_onehot)
    local N, C = M.shape(logits)
    local maxes = {}
    for i=1,N do
        local row = logits[i]
        local maxv = row[1]
        for j=2,C do if row[j] > maxv then maxv = row[j] end end
        maxes[i] = maxv
    end
    local exps = M.zeros(N, C)
    for i=1,N do
        local denom = 0.0
        for j=1,C do
            local e = math.exp(logits[i][j] - maxes[i])
            exps[i][j] = e
            denom = denom + e
        end
        for j=1,C do exps[i][j] = exps[i][j] / denom end
    end
    self.probs = exps
    self.targets = target_onehot
    local loss = 0.0
    for i=1,N do
        for j=1,C do
            if target_onehot[i][j] > 0.0 then
                local p = math.max(self.probs[i][j], 1e-12)
                loss = loss - math.log(p)
            end
        end
    end
    return loss / N
end
function SoftmaxCrossEntropy:backward()
    local N, C = M.shape(self.probs)
    local dlogits = M.sub(self.probs, self.targets)
    local invN = 1.0 / N
    return M.scalar_mul(dlogits, invN)
end

local MSE = setmetatable({}, Loss)
MSE.__index = MSE
function MSE:new() return setmetatable({ type="MSE" }, MSE) end
function MSE:forward(pred, target)
    local N, D = M.shape(pred)
    self.pred = pred; self.target = target
    local diff = M.sub(pred, target)
    local sse = 0.0
    for i=1,N do for j=1,D do sse = sse + diff[i][j]^2 end end
    return sse / (N * D)
end
function MSE:backward()
    local N, D = M.shape(self.pred)
    local diff = M.sub(self.pred, self.target)
    return M.scalar_mul(diff, 2.0/(N*D))
end

------------------------------------------------------------
-- Network
------------------------------------------------------------
local Net = {}
Net.__index = Net

function Net:new()
    return setmetatable({ layers = {} }, Net)
end

function Net:add(layer)
    table.insert(self.layers, layer)
end

function Net:forward(x)
    local out = x
    for _,L in ipairs(self.layers) do
        out = L:forward(out)
    end
    return out
end

function Net:backward(grad)
    local dout = grad
    for i=#self.layers,1,-1 do
        dout = self.layers[i]:backward(dout)
    end
    return dout
end

function Net:_iter_params()
    local params = {}
    for _,L in ipairs(self.layers) do
        if L.type == "Linear" then
            table.insert(params, {W=L.W, dW=L.dW, mW=L.mW, vW=L.vW})
            table.insert(params, {W=L.b, dW=L.db, mW=L.mB, vW=L.vB, is_bias=true})
        end
    end
    return params
end

function Net:zero_grad()
    for _,L in ipairs(self.layers) do
        if L.type == "Linear" then
            local n,m = M.shape(L.dW)
            for i=1,n do for j=1,m do L.dW[i][j] = 0.0 end end
            local n2,m2 = M.shape(L.db)
            for i=1,n2 do for j=1,m2 do L.db[i][j] = 0.0 end end
        end
    end
end

local function sgd_update(P, lr)
    local W, dW = P.W, P.dW
    local n,m = M.shape(W)
    for i=1,n do
        for j=1,m do
            W[i][j] = W[i][j] - lr * dW[i][j]
        end
    end
end

local function momentum_update(P, lr, mu)
    mu = mu or 0.9
    local W, dW = P.W, P.dW
    local V = P.mW
    local n,m = M.shape(W)
    for i=1,n do
        for j=1,m do
            V[i][j] = mu * V[i][j] - lr * dW[i][j]
            W[i][j] = W[i][j] + V[i][j]
        end
    end
end

local function adam_update(P, lr, t, beta1, beta2, eps)
    beta1 = beta1 or 0.9
    beta2 = beta2 or 0.999
    eps   = eps   or 1e-8
    local W, dW, m, v = P.W, P.dW, P.mW, P.vW
    local n,mcols = M.shape(W)
    for i=1,n do
        for j=1,mcols do
            m[i][j] = beta1 * m[i][j] + (1-beta1) * dW[i][j]
            v[i][j] = beta2 * v[i][j] + (1-beta2) * (dW[i][j]^2)
            local mhat = m[i][j] / (1 - beta1^t)
            local vhat = v[i][j] / (1 - beta2^t)
            W[i][j] = W[i][j] - lr * (mhat / (math.sqrt(vhat) + eps))
        end
    end
end

function Net:step(opt)
    opt = opt or {algo="adam", lr=1e-3}
    local algo = (opt.algo or "adam"):lower()
    for _,L in ipairs(self.layers) do
        if L.type == "Linear" then
            L.t = L.t + 1
            local params = {
                {W=L.W, dW=L.dW, mW=L.mW, vW=L.vW},
                {W=L.b, dW=L.db, mW=L.mB, vW=L.vB}
            }
            for _,P in ipairs(params) do
                if algo == "sgd" then sgd_update(P, opt.lr or 1e-2)
                elseif algo == "momentum" then momentum_update(P, opt.lr or 1e-2, opt.mu or 0.9)
                else adam_update(P, opt.lr or 1e-3, L.t, opt.beta1, opt.beta2, opt.eps)
                end
            end
        end
    end
end

------------------------------------------------------------
-- Training helpers
------------------------------------------------------------
local function shuffle_indices(n)
    local idx = {}
    for i=1,n do idx[i] = i end
    for i=n,2,-1 do
        local j = math.floor(RNG:rand() * i) + 1
        idx[i], idx[j] = idx[j], idx[i]
    end
    return idx
end

local function slice_rows(A, idxs)
    local B = {}
    for i=1,#idxs do
        B[i] = A[idxs[i]]
    end
    return B
end

local function is_vector_ints(y)
    return type(y) == 'table' and type(y[1]) == 'number' and (not type(y[1]) == 'table')
end

local function ensure_targets(y, num_classes)
    if type(y[1]) == 'table' then
        return y, #y[1]
    else
        local C = num_classes or 0
        if C == 0 then
            for i=1,#y do if y[i] > C then C = y[i] end end
        end
        return M.one_hot(y, C), C
    end
end

function Net:fit(X, y, cfg)
    cfg = cfg or {}
    local epochs = cfg.epochs or 50
    local batch = cfg.batch or 16
    local lr    = cfg.lr or 1e-3
    local algo  = cfg.algo or 'adam'
    local verbose = (cfg.verbose ~= false)
    local task = cfg.task or 'auto' -- 'auto' | 'classification' | 'regression'

    local N, Din = M.shape(X)
    local Dout

    local criterion
    local targets

    if task == 'regression' then
        criterion = MSE:new()
        targets = y
        local _, d = M.shape(y)
        Dout = d
    else
        targets, Dout = ensure_targets(y, cfg.num_classes)
        criterion = SoftmaxCrossEntropy:new()
    end

    local num_batches = math.max(1, math.floor((N + batch - 1) / batch))

    for epoch=1,epochs do
        local idx = shuffle_indices(N)
        local epoch_loss = 0.0

        for b=1,num_batches do
            local s = (b-1)*batch + 1
            local e = math.min(b*batch, N)
            if s > e then break end
            local sel = {}
            for i=s,e do sel[#sel+1] = idx[i] end
            local xbatch = slice_rows(X, sel)
            local ybatch
            if criterion.type == 'SoftmaxCrossEntropy' then
                ybatch = slice_rows(targets, sel)
            else
                ybatch = slice_rows(targets, sel)
            end

            -- forward
            local logits = self:forward(xbatch)
            local loss = criterion:forward(logits, ybatch)
            epoch_loss = epoch_loss + loss

            -- backward
            local dloss = criterion:backward()
            self:backward(dloss)

            -- step
            self:step({algo=algo, lr=lr, beta1=cfg.beta1, beta2=cfg.beta2, eps=cfg.eps, mu=cfg.mu})
            self:zero_grad()
        end

        epoch_loss = epoch_loss / num_batches

        if verbose then
            if criterion.type == 'SoftmaxCrossEntropy' then
                local preds = self:predict_classes(X)
                local acc = NN.metrics.accuracy_from_indices(preds, y)
                printf("epoch %d | loss %.6f | acc %.2f%%", epoch, epoch_loss, 100*acc)
            else
                printf("epoch %d | loss %.6f", epoch, epoch_loss)
            end
        end
    end
end

function Net:predict_proba(X)
    local logits = self:forward(X)
    local N,C = M.shape(logits)
    local probs = M.zeros(N,C)
    for i=1,N do
        local row = logits[i]
        local maxv = row[1]
        for j=2,C do if row[j] > maxv then maxv = row[j] end end
        local denom = 0.0
        for j=1,C do
            probs[i][j] = math.exp(row[j]-maxv)
            denom = denom + probs[i][j]
        end
        for j=1,C do probs[i][j] = probs[i][j] / denom end
    end
    return probs
end

function Net:predict_classes(X)
    local probs = self:predict_proba(X)
    return M.argmax_rowwise(probs)
end

------------------------------------------------------------
-- Metrics
------------------------------------------------------------
NN.metrics = {}
function NN.metrics.accuracy_from_indices(pred, y)
    local n = #pred
    local correct = 0
    if type(y[1]) == 'table' then
        local yi = {}
        for i=1,n do
            local row = y[i]
            local maxv, idx = row[1], 1
            for j=2,#row do if row[j] > maxv then maxv, idx = row[j], j end end
            yi[i] = idx
        end
        y = yi
    end
    for i=1,n do if pred[i] == y[i] then correct = correct + 1 end end
    return correct / n
end

function NN.metrics.confusion_matrix(pred, y, num_classes)
    if type(y[1]) == 'table' then
        y = M.argmax_rowwise(y)
    end
    local C = num_classes or 0
    for i=1,#y do if y[i] > C then C = y[i] end end
    for i=1,#pred do if pred[i] > C then C = pred[i] end end
    local mat = M.zeros(C, C)
    for i=1,#y do
        mat[y[i]][pred[i]] = mat[y[i]][pred[i]] + 1
    end
    return mat
end

------------------------------------------------------------
-- Weights
------------------------------------------------------------
local function _is_ident(k)
    return type(k) == 'string' and k:match('^[_%a][_%w]*$') ~= nil
end

function _Serialize(val)
    local t = type(val)
    if t == 'number' then
        return string.format('%.17g', val)
    elseif t == 'string' then
        return string.format('%q', val)
    elseif t == 'table' then
        local n = #val
        local count, arrayish = 0, true
        for k,_ in pairs(val) do
            count = count + 1
            if type(k) ~= 'number' then arrayish = false end
        end
        if arrayish and count == n then
            local parts = {}
            for i=1,n do parts[i] = _Serialize(val[i]) end
            return '{'..table.concat(parts, ',')..'}'
        else
            local parts = {}
            for k,v in pairs(val) do
                local key
                if _is_ident(k) then key = k .. '=' else key = '['.._Serialize(k)..']=' end
                parts[#parts+1] = key .. _Serialize(v)
            end
            return '{'..table.concat(parts, ',')..'}'
        end
    else
        return 'nil'
    end
end

function Save_weights(net, path)
    local state = NN.save_state(net)
    local f, err = io.open(path, 'w')
    assert(f, 'save_weights: '..tostring(err))
    f:write('return ')
    f:write(_Serialize(state))
    f:write('\n')
    f:close()
end

function Load_weights(net, path)
    local chunk, err = loadfile(path)
    assert(chunk, 'load_weights: '..tostring(err))
    local ok, state = pcall(chunk)
    assert(ok and type(state) == 'table', 'load_weights: file did not return a table')
    NN.load_state(net, state)
end

function Dumps_weights(net)
    local state = NN.save_state(net)
    return _Serialize(state)
end

function Loads_weights(net, s)
    local loader = load or loadstring       -- 5.1/5.2/5.3 compat
    local chunk, err = loader('return '..s)
    assert(chunk, 'loads_weights: '..tostring(err))
    local ok, state = pcall(chunk)
    assert(ok and type(state) == 'table', 'loads_weights: string did not evaluate to table')
    NN.load_state(net, state)
end


------------------------------------------------------------
-- Public API
------------------------------------------------------------
NN.new = function() return Net:new() end
NN.Linear  = function(in_dim, out_dim, opts) return Linear:new(in_dim, out_dim, opts) end
NN.ReLU    = function() return ReLU:new() end
NN.Sigmoid = function() return Sigmoid:new() end
NN.Tanh    = function() return Tanh:new() end
NN.SoftmaxCrossEntropy = function() return SoftmaxCrossEntropy:new() end
NN.MSE = function() return MSE:new() end
-- NN.M = NN.backend
NN.M = M
NN.RNG = RNG
NN.save_weights = function(net, path) Save_weights(net, path) end
NN.load_weights = function(net, path) Load_weights(net, path) end
NN.dumps_weights = function(net) return Dumps_weights(net) end
NN.loads_weights = function(net, s) Loads_weights(net, s) end

function NN.save_state(net)
    local state = {}
    for li,L in ipairs(net.layers) do
        if L.type == 'Linear' then
        state[#state+1] = {type='Linear', W=M.copy(L.W), b=M.copy(L.b)}
        else
        state[#state+1] = {type=L.type}
        end
    end
    return state
end

function NN.load_state(net, state)
    for i,entry in ipairs(state) do
        local L = net.layers[i]
        if L and L.type == 'Linear' and entry.type == 'Linear' then
        L.W = M.copy(entry.W)
        L.b = M.copy(entry.b)
        end
    end
end

------------------------------------------------------------
-- XOR Demo
------------------------------------------------------------
local function make_xor(n)
    local X = M.zeros(n, 2)
    local y = {}
    for i=1,n do
        local a = (RNG:rand() < 0.5) and 0 or 1
        local b = (RNG:rand() < 0.5) and 0 or 1
        X[i][1] = a
        X[i][2] = b
        local label = (imod(a + b, 2) == 1) and 2 or 1
        y[i] = label
    end
    return X, y
end


local function run_xor_demo()
    printf("\n== XOR demo ==")
    RNG:randomseed(123)
    local X, y = make_xor(400)
    local net = NN.new()
    net:add(NN.Linear(2, 8))
    net:add(NN.Tanh())
    net:add(NN.Linear(8, 2))

    net:fit(X, y, {epochs=200, batch=32, lr=0.05, algo='adam', verbose=true})

    local preds = net:predict_classes(X)
    local acc = NN.metrics.accuracy_from_indices(preds, y)
    printf("Final training accuracy: %.2f%%", acc*100)

    local cm = NN.metrics.confusion_matrix(preds, y, 2)
    printf("Confusion Matrix (rows=true, cols=pred):")
    for i=1,#cm do
        local row = cm[i]
        printf("  %d  %d", row[1], row[2])
    end

    local test = { {0,0}, {0,1}, {1,0}, {1,1} }
    local proba = net:predict_proba(test)
    printf("\nProbs for [0,0],[0,1],[1,0],[1,1]: (class1, class2)")
    for i=1,#proba do printf("  %.3f  %.3f", proba[i][1], proba[i][2]) end
end

if arg and arg[1] == 'demo' then
    run_xor_demo()
end

return NN







# Lua Neural Network Library – Function Reference

This document describes the public functions of the pure-Lua neural network module.  
It assumes the file is saved as `nn.lua` and required as:

```lua
local NN = require("nn")
```

The library exposes:

- **High-level network construction helpers** (`NN.new`, `NN.Linear`, `NN.ReLU`, …)
- **Training and prediction methods** on the network object (`net:fit`, `net:predict_proba`, …)
- **Matrix utilities** (`NN.M.*`)
- **Random number utilities** (`NN.RNG.*`)
- **Metrics helpers** (`NN.metrics.*`)
- **Save/load helpers** for weights and state

This file focuses on *functions and methods only*.

---

## Compute device selection

Use the unified module for both CPU and GPU compute:

```lua
local NN = require("nn")
```

For a CPU-only single-file setup, you can still import the standalone CPU
module:

```lua
local NN = require("nn_cpu")
```

That path does not load GPU routing, native bridge checks, or GPU warnings.

CPU is the default. Pass `device` options to high-level network calls when you
want to choose explicitly:

```lua
net:fit(X, y, { device = "cpu" })
net:fit(X, y, { device = "gpu" })
local logits = net:forward(X, { device = "gpu" })
local probs = net:predict_proba(X, { device = "gpu" })
local classes = net:predict_classes(X, { device = "gpu" })
```

You can also set a default device when creating a network:

```lua
local net = NN.new({ device = "auto" })
```

Per-call `device` options override the network default.

Supported device values are:

- `"cpu"` – always use the pure-Lua backend.
- `"gpu"` – request OpenCL GPU execution.
- `"auto"` – use GPU when available, otherwise CPU.

If GPU support is unavailable, the module warns once and falls back to CPU.
Set `gpu_required = true` to raise an error instead:

```lua
net:fit(X, y, { device = "gpu", gpu_required = true })
```

GPU status helpers:

- `NN.gpu_available()`
- `NN.gpu_info()`
- `NN.available_devices()`

LuaJIT can use the embedded OpenCL backend directly through FFI. Plain Lua
5.4 can load the vendored native bridge and probe OpenCL, but it cannot execute
GPU tensor kernels yet because the bridge tensor entrypoints are not implemented.
On plain Lua 5.4, `device = "gpu"` therefore falls back to CPU unless
`gpu_required = true` is set. Use LuaJIT when you need GPU compute today.

The lower-level `nn_cpu` and `nn_gpu_backend` modules remain available for
compatibility, but new code should prefer `require("nn")`.

---

## 1. Core module API (`NN`)

### `NN.new(opts)`

Create a new, empty sequential network.

```lua
local net = NN.new()
local gpu_preferring_net = NN.new({ device = "auto" })
```

**Arguments**

- `opts` (table, optional):
  - `device` (`"cpu"`, `"gpu"`, or `"auto"`, optional) – default compute
    device for high-level network calls. Defaults to `"cpu"`.
  - `gpu_required` (boolean, optional) – if true, GPU requests error instead
    of falling back to CPU.

**Returns**

- `net` – a network object with methods described in [Section 2](#2-network-methods).

You typically populate it with layers:

```lua
net:add(NN.Linear(4, 16))
net:add(NN.ReLU())
net:add(NN.Linear(16, 3))
```

---

### `NN.Linear(in_dim, out_dim, opts)`

Create a fully-connected (dense) linear layer:

```lua
local layer = NN.Linear(10, 5, { init_scale = 0.1 })
net:add(layer)
```

**Arguments**

- `in_dim` (integer) – input dimensionality (number of features).
- `out_dim` (integer) – output dimensionality (number of units).
- `opts` (table, optional):
  - `init_scale` (number, optional) – scaling factor for random weight
    initialization. Defaults to a value based on `sqrt(2 / in_dim)`.

**Returns**

- A **layer instance** that can be added to a network via `net:add(layer)`.

**Input / output shapes**

- Input to `:forward` should be a matrix of shape **N × in_dim**.
- Output is a matrix of shape **N × out_dim**.

---

### `NN.ReLU()`

Create a ReLU activation layer.

```lua
net:add(NN.ReLU())
```

**Returns**

- A ReLU activation layer that can be added after a linear layer.

**Effect**

- Applies `max(0, x)` elementwise.

---

### `NN.Sigmoid()`

Create a Sigmoid activation layer.

```lua
net:add(NN.Sigmoid())
```

**Returns**

- A Sigmoid activation layer.

**Effect**

- Applies `σ(x) = 1/(1+e^{-x})` elementwise.

---

### `NN.Tanh()`

Create a Tanh activation layer.

```lua
net:add(NN.Tanh())
```

**Returns**

- A Tanh activation layer.

**Effect**

- Applies `tanh(x)` elementwise.

---

### `NN.SoftmaxCrossEntropy()`

Create a softmax + cross-entropy loss object (for classification).

```lua
local criterion = NN.SoftmaxCrossEntropy()
local loss = criterion:forward(logits, targets_one_hot)
local dlogits = criterion:backward()
```

This is mainly used *internally* by `net:fit` when doing classification, but
it is also exposed for manual training loops.

**Returns**

- A loss object with methods:
  - `:forward(logits, target_onehot)`
  - `:backward()`

See [Loss objects](#4-loss-objects) for details.

---

### `NN.MSE()`

Create a mean squared error (MSE) loss object (for regression).

```lua
local criterion = NN.MSE()
local loss = criterion:forward(pred, target)
local dpred = criterion:backward()
```

Also used internally by `net:fit` for regression.

**Returns**

- A loss object with methods:
  - `:forward(pred, target)`
  - `:backward()`

See [Loss objects](#4-loss-objects) for details.

---

### `NN.M` – Matrix utilities

`NN.M` is a table of matrix helper functions used throughout the library.

All matrices are represented as **tables of rows**:

```lua
-- 2 × 3 matrix
local A = {
  {1.0, 2.0, 3.0},
  {4.0, 5.0, 6.0},
}
```

Key functions:

- [`NN.M.shape(A)`](#nnmshapea)
- [`NN.M.zeros(n, m)`](#nnmzerosn-m)
- [`NN.M.ones(n-m)`](#nnmonesn-m)
- [`NN.M.randn(n-m-scale)`](#nnmrandnn-m-scale)
- [`NN.M.copy(A)`](#nnmcopya)
- [`NN.M.transpose(A)`](#nnmtransposea)
- [`NN.M.matmul(A-b)`](#nnmmatmula-b)
- [`NN.M.add(A-b)`](#nnmadda-b)
- [`NN.M.sub(A-b)`](#nnmsuba-b)
- [`NN.M.hadamard(A-b)`](#nnmhadamarda-b)
- [`NN.M.scalar_add(A-s)`](#nnmscalar_adda-s)
- [`NN.M.scalar_mul(A-s)`](#nnmscalar_mula-s)
- [`NN.M.sum(A-axis)`](#nnmsuma-axis)
- [`NN.M.apply(A-fn)`](#nnmapplya-fn)
- [`NN.M.argmax_rowwise(A)`](#nnmargmax_rowwisea)
- [`NN.M.one_hot(y-num_classes)`](#nnmone_hoty-num_classes)

(Full details in [Section 5](#5-matrix-backend-nnm).)

---

### `NN.RNG` – Random number generator

`NN.RNG` exposes a simple PRNG used internally, but you can also use it directly.

**Functions**

- [`NN.RNG:randomseed(s)`](#nnrngrandomseeds)
- [`NN.RNG:rand()`](#nnrngrand)
- [`NN.RNG:randn()`](#nnrngrandn)

See [Section 6](#6-random-number-generator-nnrng) for details.

---

### Weight and state helpers

#### `NN.save_weights(net, path)`

Save a network’s weights to a Lua file.

```lua
NN.save_weights(net, "weights.lua")
```

**Arguments**

- `net` – a network instance (from `NN.new()`).
- `path` (string) – file path to write. The file contains a Lua expression and
  can be loaded with `loadfile`.

The file structure is:

```lua
return { ... state table ... }
```

---

#### `NN.load_weights(net, path)`

Load network weights from a file created by `NN.save_weights`.

```lua
NN.load_weights(net, "weights.lua")
```

**Arguments**

- `net` – a network instance whose layers should receive loaded weights.
- `path` (string) – filename created by `NN.save_weights`.

Only `Linear` layers’ weights and biases are restored.

---

#### `NN.dumps_weights(net)`

Serialize a network’s weights to a string.

```lua
local s = NN.dumps_weights(net)
-- you can store this string or send it over the network
```

**Arguments**

- `net` – a network instance.

**Returns**

- `s` (string) – a Lua expression that evaluates to the weight/state table.

---

#### `NN.loads_weights(net, s)`

Load weights from a string created by `NN.dumps_weights`.

```lua
NN.loads_weights(net, s)
```

**Arguments**

- `net` – network instance to modify.
- `s` (string) – serialized state.

---

### State helpers (table-based)

#### `NN.save_state(net)`

Return a Lua table representing the **weights only** of a network.

```lua
local state = NN.save_state(net)
-- state is a plain Lua table you can store yourself (e.g. JSON, msgpack, etc.)
```

**Arguments**

- `net` – network instance.

**Returns**

- `state` (table) – array of per-layer entries:
  - For `Linear` layers: `{ type = "Linear", W = <matrix>, b = <matrix> }`
  - For other layers: `{ type = <string> }`

---

#### `NN.load_state(net, state)`

Load weights from a state table.

```lua
NN.load_state(net, state)
```

**Arguments**

- `net` – network instance to receive the weights.
- `state` (table) – format returned by `NN.save_state`.

Only matching `Linear` layers get their `W` and `b` replaced.

---

## 2. Network methods

A network is created with `NN.new()` and then configured with layers.

```lua
local net = NN.new()
net:add(NN.Linear(2, 8))
net:add(NN.Tanh())
net:add(NN.Linear(8, 2))
```

### `net:add(layer)`

Add a layer to the sequential network.

```lua
net:add(NN.ReLU())
net:add(NN.Linear(16, 10))
```

**Arguments**

- `layer` – any layer object created via `NN.Linear`, `NN.ReLU`, `NN.Sigmoid`,
  or `NN.Tanh`.

Layers are appended in order and used during forward and backward passes.

---

### `net:forward(X)`

Compute the forward pass through all layers.

```lua
local logits = net:forward(X)
```

**Arguments**

- `X` – input data matrix of shape **N × D**:
  - `N` – number of samples (rows).
  - `D` – input dimension, matching the first `Linear` layer’s `in_dim`.

**Returns**

- Output matrix after applying all layers in sequence. Shape depends on the
  last layer (e.g. `N × num_classes` for classification).

---

### `net:backward(grad)`

Backpropagate a gradient from the network output to its input.

```lua
local dX = net:backward(dOut)
```

**Arguments**

- `grad` – gradient with respect to network output. Shape should match the
  result of `net:forward`.

**Returns**

- Gradient with respect to the network input (`dX`).

This method also accumulates weight/bias gradients inside `Linear` layers.

---

### `net:zero_grad()`

Reset stored gradients of all `Linear` layers to zero.

```lua
net:zero_grad()
```

Use this between optimization steps if you are writing a manual training loop.

---

### `net:step(opt)`

Apply an optimizer update step to all learnable parameters.

```lua
net:step{
  algo  = "adam",  -- "adam" (default), "sgd", or "momentum"
  lr    = 1e-3,
  mu    = 0.9,     -- for momentum (optional)
  beta1 = 0.9,     -- for Adam (optional)
  beta2 = 0.999,   -- for Adam (optional)
  eps   = 1e-8,    -- for Adam (optional)
}
```

**Arguments**

- `opt` (table, optional):
  - `algo` (string) – `"adam"` (default), `"sgd"`, or `"momentum"`.
  - `lr` (number) – learning rate.
    - For `"adam"`, default is `1e-3`.
    - For `"sgd"` / `"momentum"`, default is `1e-2`.
  - `mu` (number, optional) – momentum factor (used by `"momentum"`).
  - `beta1`, `beta2`, `eps` – Adam hyperparameters (optional).

**Effect**

- Updates all `Linear` layers’ weight and bias parameters in-place, using the
  chosen optimization algorithm and accumulated gradients.

---

### `net:fit(X, y, cfg)`

High-level training function with mini-batches and shuffling.

```lua
local net = NN.new()
net:add(NN.Linear(2, 8))
net:add(NN.Tanh())
net:add(NN.Linear(8, 2))

net:fit(X, y, {
  epochs  = 200,
  batch   = 32,
  lr      = 0.05,
  algo    = "adam",
  task    = "classification",  -- or "regression" or "auto"
  verbose = true,
})
```

**Arguments**

- `X` – input data matrix, shape **N × D**.
- `y` – target labels:
  - For **classification**:
    - Either class indices (1-based), e.g. `{1, 2, 1, ...}`.
    - Or one-hot matrix `N × C`.
  - For **regression**:
    - A matrix of shape `N × Dout` with continuous targets.
- `cfg` (table, optional):
  - `epochs` (integer, default `50`) – number of passes over the dataset.
  - `batch` (integer, default `16`) – mini-batch size.
  - `lr` (number, default `1e-3`) – learning rate.
  - `algo` (string, default `"adam"`) – optimizer (`"adam"`, `"sgd"`, `"momentum"`).
  - `verbose` (boolean, default `true`) – print progress after each epoch.
  - `task` (string, default `"auto"`) – `"classification"`, `"regression"`, or `"auto"`:
    - `"regression"` – use MSE loss.
    - otherwise – use Softmax+CrossEntropy with class targets.
  - `num_classes` (integer, optional) – number of classes (if not provided,
    inferred from `y` for classification tasks).
  - `beta1`, `beta2`, `eps`, `mu` – optimizer hyperparameters passed to `net:step`.

**Behavior**

- Shuffles sample indices every epoch.
- Splits into mini-batches.
- For each mini-batch:
  - Runs forward pass.
  - Computes loss.
  - Runs backward pass.
  - Calls `net:step` and `net:zero_grad`.
- At the end of each epoch:
  - Computes average loss over batches.
  - For classification:
    - Computes accuracy on the *full* dataset via
      `NN.metrics.accuracy_from_indices(net:predict_classes(X), y)`.
  - Optionally prints `epoch`, `loss`, and `acc`.

---

### `net:predict_proba(X)`

Compute class probabilities for a batch of inputs (classification).

```lua
local probs = net:predict_proba(X)
-- probs is N × C, rows sum to 1
```

**Arguments**

- `X` – input matrix `N × D`.

**Returns**

- `probs` – matrix `N × C` containing softmax probabilities over classes.

Use this only for classification-style networks where the final layer produces
class logits.

---

### `net:predict_classes(X)`

Compute predicted class indices (argmax of probabilities).

```lua
local pred = net:predict_classes(X)
```

**Arguments**

- `X` – input matrix `N × D`.

**Returns**

- `pred` – table of length `N` where `pred[i]` is the predicted class index
  (1-based).

Internally uses `net:predict_proba` and `NN.M.argmax_rowwise`.

---

## 3. Layer objects

Layers are typically created via `NN.Linear`, `NN.ReLU`, `NN.Sigmoid`, `NN.Tanh`
and then added to a network. They also expose their own `:forward` and
`:backward` methods which are used by the network.

### Linear layer

Created by:

```lua
local L = NN.Linear(in_dim, out_dim, opts)
```

**Methods**

- `L:forward(X)` – forward pass:
  - Input: `X` (matrix `N × in_dim`).
  - Output: `Y` (matrix `N × out_dim`).
- `L:backward(dY)` – backward pass:
  - Input: `dY` (matrix `N × out_dim`).
  - Output: `dX` (matrix `N × in_dim`).

The layer keeps internal fields for weights (`W`), biases (`b`), gradients (`dW`,
`db`), and optimizer state (`mW`, `vW`, `mB`, `vB`).

---

### ReLU layer

Created by:

```lua
local relu = NN.ReLU()
```

**Methods**

- `relu:forward(X)` – elementwise `max(0, x)`.
- `relu:backward(dY)` – multiplies gradient by a mask where `X > 0`.

---

### Sigmoid layer

Created by:

```lua
local sig = NN.Sigmoid()
```

**Methods**

- `sig:forward(X)` – elementwise sigmoid.
- `sig:backward(dY)` – uses derivative `y * (1 - y)` where `y` is the sigmoid output.

---

### Tanh layer

Created by:

```lua
local t = NN.Tanh()
```

**Methods**

- `t:forward(X)` – elementwise hyperbolic tangent.
- `t:backward(dY)` – uses derivative `1 - tanh(x)^2` (in terms of outputs).

---

## 4. Loss objects

### `SoftmaxCrossEntropy` (classification)

Construct with:

```lua
local crit = NN.SoftmaxCrossEntropy()
```

**Methods**

#### `crit:forward(logits, target_onehot)`

```lua
local loss = crit:forward(logits, targets)
```

**Arguments**

- `logits` – matrix `N × C` (raw class scores from the net).
- `target_onehot` – one-hot matrix `N × C` of target classes.

**Returns**

- `loss` (number) – average cross entropy over the batch.

Internally stores softmax probabilities and targets for use in `:backward()`.

#### `crit:backward()`

```lua
local dlogits = crit:backward()
```

**Returns**

- `dlogits` – gradient of loss w.r.t. `logits` (`N × C`).

---

### `MSE` (regression)

Construct with:

```lua
local crit = NN.MSE()
```

**Methods**

#### `crit:forward(pred, target)`

```lua
local loss = crit:forward(pred, target)
```

**Arguments**

- `pred` – predictions matrix `N × D`.
- `target` – targets matrix `N × D`.

**Returns**

- `loss` – mean squared error over all elements.

#### `crit:backward()`

```lua
local dpred = crit:backward()
```

**Returns**

- `dpred` – gradient of loss w.r.t. `pred` (`N × D`).

---

## 5. Matrix backend (`NN.M`)

All functions operate on row-major matrices (tables of row tables).

### `NN.M.shape(A)`

```lua
local n, m = NN.M.shape(A)
```

**Returns**

- `n` – number of rows.
- `m` – number of columns (0 if `A` is empty).

---

### `NN.M.zeros(n, m)`

Create an `n × m` matrix filled with `0.0`.

```lua
local Z = NN.M.zeros(3, 4)
```

---

### `NN.M.ones(n, m)`

Create an `n × m` matrix filled with `1.0`.

```lua
local O = NN.M.ones(2, 2)
```

---

### `NN.M.randn(n, m, scale)`

Create an `n × m` matrix with entries drawn from `N(0, scale^2)`.

```lua
local W = NN.M.randn(5, 5, 0.1)
```

**Arguments**

- `n`, `m` – dimensions.
- `scale` (number, optional, default `1.0`) – standard deviation scaling.

---

### `NN.M.copy(A)`

Deep copy a matrix.

```lua
local B = NN.M.copy(A)
```

---

### `NN.M.transpose(A)`

Transpose a matrix.

```lua
local AT = NN.M.transpose(A)
```

**Input**

- `A` – `n × m` matrix.

**Returns**

- `AT` – `m × n` matrix.

---

### `NN.M.matmul(A, B)`

Matrix multiplication.

```lua
local C = NN.M.matmul(A, B)
```

**Arguments**

- `A` – shape `n × k`.
- `B` – shape `k × p`.

**Returns**

- `C` – shape `n × p`.

Asserts on incompatible dimensions.

---

### `NN.M.add(A, B)`

Elementwise addition with optional row-wise broadcasting.

```lua
local C = NN.M.add(A, B)
```

**Arguments**

- `A` – `n × m` matrix.
- `B` – either:
  - `n × m` matrix, or
  - `1 × m` matrix (broadcast across rows).

**Returns**

- `C` – `n × m` matrix.

---

### `NN.M.sub(A, B)`

Elementwise subtraction with optional row-wise broadcasting.

```lua
local C = NN.M.sub(A, B)
```

Same shape rules as `NN.M.add`.

---

### `NN.M.hadamard(A, B)`

Elementwise multiplication (Hadamard product).

```lua
local C = NN.M.hadamard(A, B)
```

**Arguments**

- `A`, `B` – same shape `n × m`.

---

### `NN.M.scalar_add(A, s)`

Add scalar `s` to every element of `A`.

```lua
local B = NN.M.scalar_add(A, 1.0)
```

---

### `NN.M.scalar_mul(A, s)`

Multiply every element by scalar `s`.

```lua
local B = NN.M.scalar_mul(A, 0.5)
```

---

### `NN.M.sum(A, axis)`

Sum elements over different axes.

```lua
local total = NN.M.sum(A)        -- scalar
local colsum = NN.M.sum(A, 1)    -- 1 × m
local rowsum = NN.M.sum(A, 2)    -- n × 1
```

**Arguments**

- `A` – `n × m` matrix.
- `axis` (optional):
  - `nil` – sum all elements, return scalar.
  - `1` – sum over rows -> returns matrix `1 × m` (column sums).
  - `2` – sum over columns -> returns matrix `n × 1` (row sums).

---

### `NN.M.apply(A, fn)`

Apply a scalar function elementwise.

```lua
local B = NN.M.apply(A, math.abs)
```

**Arguments**

- `A` – matrix.
- `fn` – function taking a number and returning a number.

**Returns**

- New matrix with `fn` applied to each element.

---

### `NN.M.argmax_rowwise(A)`

Row-wise argmax.

```lua
local idx = NN.M.argmax_rowwise(A)
```

**Arguments**

- `A` – `n × m` matrix.

**Returns**

- `idx` – table of length `n`, where `idx[i]` is index (1-based) of the max
  element in row `i`.

---

### `NN.M.one_hot(y, num_classes)`

Convert a vector of class indices to a one-hot matrix.

```lua
local Y = NN.M.one_hot({1, 3, 2}, 3)
```

**Arguments**

- `y` – table of length `n`, each entry in `1..num_classes`.
- `num_classes` (integer) – number of classes.

**Returns**

- `Y` – `n × num_classes` one-hot matrix.

---

## 6. Random number generator (`NN.RNG`)

### `NN.RNG:randomseed(s)`

Seed the internal PRNG.

```lua
NN.RNG:randomseed(1234)
```

**Arguments**

- `s` (number, optional) – seed value (defaults to `os.time()` if not provided).

---

### `NN.RNG:rand()`

Generate a uniform random number in `[0,1)`.

```lua
local u = NN.RNG:rand()
```

---

### `NN.RNG:randn()`

Generate a random number approximately distributed as `N(0,1)`.

```lua
local z = NN.RNG:randn()
```

Uses the Box–Muller transform internally.

---

## 7. Metrics (`NN.metrics`)

### `NN.metrics.accuracy_from_indices(pred, y)`

Compute classification accuracy from predicted and true labels.

```lua
local acc = NN.metrics.accuracy_from_indices(pred, y)
print("Accuracy:", acc)
```

**Arguments**

- `pred` – table of length `n` with predicted class indices (1-based).
- `y` – true labels:
  - Either class indices same shape as `pred`, or
  - One-hot matrix `n × C`.

**Returns**

- `acc` – accuracy in `[0,1]` (fraction of correct predictions).

---

### `NN.metrics.confusion_matrix(pred, y, num_classes)`

Compute a confusion matrix.

```lua
local cm = NN.metrics.confusion_matrix(pred, y, num_classes)
```

**Arguments**

- `pred` – predicted class indices (as in `accuracy_from_indices`).
- `y` – true labels (indices or one-hot).
- `num_classes` (integer, optional) – number of classes; if omitted, inferred
  from `pred` and `y`.

**Returns**

- `mat` – `C × C` matrix where `mat[true][pred]` counts how many samples of
  true class `true` were predicted as class `pred`.

---

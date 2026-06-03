## See docs/nn-usage.md for instructions

### Usage

Import the unified module:

```lua
local NN = require("nn")

local net = NN.new()
net:add(NN.Linear(2, 8))
net:add(NN.ReLU())
net:add(NN.Linear(8, 2))

net:fit(X, y, { epochs = 50, batch = 16 }) -- CPU by default
local probs = net:predict_proba(X)
```

For a CPU-only single-file setup, you can still import the standalone module:

```lua
local NN = require("nn_cpu")
```

That path does not load GPU routing, native bridge checks, or GPU warnings.

Switch compute device per call with `device = "cpu"`, `"gpu"`, or `"auto"`:

```lua
local net = NN.new({ device = "auto" }) -- default for this network
net:fit(X, y, { epochs = 50, batch = 16, device = "gpu" })
local probs = net:predict_proba(X, { device = "gpu" })
```

If GPU support is not available, the backend warns once and falls back to CPU
unless `gpu_required = true` is set.

LuaJIT can run GPU tensor compute through the embedded OpenCL FFI backend.
Plain Lua 5.4 can load the vendored bridge and probe OpenCL, but it cannot
execute GPU tensor kernels yet; `device = "gpu"` falls back to CPU unless
`gpu_required = true` is set, in which case it raises a clear error. Use
LuaJIT for GPU compute today.

### Tests

```powershell
lua tests/run_tests.lua
luajit tests/run_tests.lua
```

Or from inside the `tests/` directory:

```powershell
lua run_tests.lua
luajit run_tests.lua
```

## Use my csv lib alongside :)

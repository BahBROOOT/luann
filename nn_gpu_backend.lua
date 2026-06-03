------------------------------------------------------------
-- Init
-- See docs/nn-usage.md for instructions.
-- Use tests/run_tests.lua to run the test suite.
------------------------------------------------------------

local NN = require("nn_cpu")
local M = NN.M

local GPU = {
    warned = false,
    init_attempted = false,
    ctx = nil,
    unavailable_reason = nil,
    native_bridge = nil,
    native_probe = nil,
}

local original = {}

local function warn_once(msg)
    if GPU.warned then return end
    GPU.warned = true
    io.stderr:write("nn_gpu_backend: " .. msg .. "\n")
end

local function copy_table(t)
    local out = {}
    if t then
        for k,v in pairs(t) do out[k] = v end
    end
    return out
end

local function shape(A)
    return M.shape(A)
end

local function clear_gpu_state(net)
    if not net or not net.layers then return end
    for _,L in ipairs(net.layers) do
        L._gpu = nil
        L._gpu_cache = nil
    end
end

local function ensure_targets(y, num_classes)
    if type(y[1]) == "table" then
        return y, #y[1]
    end

    local C = num_classes or 0
    if C == 0 then
        for i=1,#y do
            if y[i] > C then C = y[i] end
        end
    end
    return M.one_hot(y, C), C
end

local function shuffle_indices(n)
    local idx = {}
    for i=1,n do idx[i] = i end
    for i=n,2,-1 do
        local j = math.floor(NN.RNG:rand() * i) + 1
        idx[i], idx[j] = idx[j], idx[i]
    end
    return idx
end

local function make_batch_indices(idx, first_i, last_i)
    local out = {}
    for i=first_i,last_i do
        out[#out+1] = idx[i] - 1
    end
    return out
end

local function platform_separator()
    return package.config:sub(1, 1)
end

local function runtime_name()
    return jit and "LuaJIT" or _VERSION
end

local function plain_lua_gpu_reason(extra)
    local msg = runtime_name()
        .. " cannot execute GPU tensor compute yet. "
        .. "GPU compute currently requires LuaJIT, which uses the embedded OpenCL FFI backend. "
        .. "Use luajit for device=\"gpu\", or use device=\"cpu\"/device=\"auto\" on plain Lua."
    if extra and extra ~= "" then
        msg = msg .. " " .. extra
    end
    return msg
end

local function host_platform()
    local sep = platform_separator()
    if sep == "\\" then return "windows-x64" end
    return "linux-x64"
end

local function lua_abi()
    if jit then return "luajit" end
    local v = _VERSION:match("(%d+%.%d+)")
    if v == "5.4" then return "lua54" end
    if v == "5.3" then return "lua53" end
    if v == "5.2" then return "lua52" end
    if v == "5.1" then return "lua51" end
    return "lua"
end

local function dirname(path)
    local sep = platform_separator()
    local p = path:match("^(.*[/\\])")
    if p then return p end
    return "." .. sep
end

local function module_dir()
    local source = debug.getinfo(1, "S").source or ""
    if source:sub(1, 1) == "@" then
        return dirname(source:sub(2))
    end
    return "." .. platform_separator()
end

local function native_bridge_path()
    local sep = platform_separator()
    local ext = sep == "\\" and ".dll" or ".so"
    return module_dir()
        .. "native" .. sep
        .. "bin" .. sep
        .. host_platform() .. sep
        .. lua_abi() .. sep
        .. "luann_opencl" .. ext
end

local function try_load_native_bridge()
    local loader = package.loadlib
    if not loader then
        return nil, "package.loadlib is unavailable"
    end

    local path = native_bridge_path()
    local open_fn = "luaopen_luann_opencl"
    local f, err = loader(path, open_fn)
    if not f then
        return nil, "vendored bridge not loaded from " .. path .. ": " .. tostring(err)
    end

    local ok, bridge = pcall(f)
    if not ok then
        return nil, "vendored bridge failed to initialize: " .. tostring(bridge)
    end
    if type(bridge) ~= "table" then
        return nil, "vendored bridge did not return a Lua table"
    end
    return bridge
end

------------------------------------------------------------
-- LuaJIT FFI OpenCL implementation
------------------------------------------------------------

local KERNEL_SOURCE = [=[
__kernel void gather_rows(
    __global const float *src,
    __global const int *idx,
    __global float *dst,
    const int rows,
    const int cols)
{
    int r = get_global_id(0);
    int c = get_global_id(1);
    if (r >= rows || c >= cols) return;
    int sr = idx[r];
    dst[r * cols + c] = src[sr * cols + c];
}

__kernel void matmul_bias(
    __global const float *A,
    __global const float *B,
    __global const float *bias,
    __global float *C,
    const int N,
    const int M,
    const int P)
{
    int r = get_global_id(0);
    int c = get_global_id(1);
    if (r >= N || c >= P) return;
    float s = 0.0f;
    for (int k = 0; k < M; ++k) {
        s += A[r * M + k] * B[k * P + c];
    }
    C[r * P + c] = s + bias[c];
}

__kernel void matmul_ta(
    __global const float *A,
    __global const float *B,
    __global float *C,
    const int N,
    const int M,
    const int P)
{
    int r = get_global_id(0);
    int c = get_global_id(1);
    if (r >= M || c >= P) return;
    float s = 0.0f;
    for (int k = 0; k < N; ++k) {
        s += A[k * M + r] * B[k * P + c];
    }
    C[r * P + c] = s;
}

__kernel void matmul_tb(
    __global const float *A,
    __global const float *B,
    __global float *C,
    const int N,
    const int P,
    const int M)
{
    int r = get_global_id(0);
    int c = get_global_id(1);
    if (r >= N || c >= M) return;
    float s = 0.0f;
    for (int k = 0; k < P; ++k) {
        s += A[r * P + k] * B[c * P + k];
    }
    C[r * M + c] = s;
}

__kernel void sum_rows(
    __global const float *A,
    __global float *out,
    const int N,
    const int M)
{
    int c = get_global_id(0);
    if (c >= M) return;
    float s = 0.0f;
    for (int r = 0; r < N; ++r) {
        s += A[r * M + c];
    }
    out[c] = s;
}

__kernel void relu_forward(
    __global const float *x,
    __global float *out,
    __global float *mask,
    const int total)
{
    int i = get_global_id(0);
    if (i >= total) return;
    float v = x[i];
    if (v > 0.0f) {
        out[i] = v;
        mask[i] = 1.0f;
    } else {
        out[i] = 0.0f;
        mask[i] = 0.0f;
    }
}

__kernel void relu_backward(
    __global const float *dout,
    __global const float *mask,
    __global float *dx,
    const int total)
{
    int i = get_global_id(0);
    if (i >= total) return;
    dx[i] = dout[i] * mask[i];
}

__kernel void sigmoid_forward(
    __global const float *x,
    __global float *out,
    const int total)
{
    int i = get_global_id(0);
    if (i >= total) return;
    out[i] = 1.0f / (1.0f + exp(-x[i]));
}

__kernel void sigmoid_backward(
    __global const float *dout,
    __global const float *y,
    __global float *dx,
    const int total)
{
    int i = get_global_id(0);
    if (i >= total) return;
    float v = y[i];
    dx[i] = dout[i] * v * (1.0f - v);
}

__kernel void tanh_forward(
    __global const float *x,
    __global float *out,
    const int total)
{
    int i = get_global_id(0);
    if (i >= total) return;
    out[i] = tanh(x[i]);
}

__kernel void tanh_backward(
    __global const float *dout,
    __global const float *y,
    __global float *dx,
    const int total)
{
    int i = get_global_id(0);
    if (i >= total) return;
    float v = y[i];
    dx[i] = dout[i] * (1.0f - v * v);
}

__kernel void softmax_ce(
    __global const float *logits,
    __global const float *target,
    __global float *probs,
    __global float *losses,
    const int N,
    const int C)
{
    int r = get_global_id(0);
    if (r >= N) return;
    int base = r * C;
    float maxv = logits[base];
    for (int c = 1; c < C; ++c) {
        float v = logits[base + c];
        if (v > maxv) maxv = v;
    }

    float denom = 0.0f;
    for (int c = 0; c < C; ++c) {
        float e = exp(logits[base + c] - maxv);
        probs[base + c] = e;
        denom += e;
    }

    float loss = 0.0f;
    for (int c = 0; c < C; ++c) {
        float p = probs[base + c] / denom;
        probs[base + c] = p;
        float t = target[base + c];
        if (t > 0.0f) {
            loss -= t * log(fmax(p, 1.0e-12f));
        }
    }
    losses[r] = loss;
}

__kernel void softmax_ce_backward(
    __global const float *probs,
    __global const float *target,
    __global float *dout,
    const float invN,
    const int total)
{
    int i = get_global_id(0);
    if (i >= total) return;
    dout[i] = (probs[i] - target[i]) * invN;
}

__kernel void mse_loss_grad(
    __global const float *pred,
    __global const float *target,
    __global float *grad,
    __global float *loss_terms,
    const float grad_scale,
    const int total)
{
    int i = get_global_id(0);
    if (i >= total) return;
    float d = pred[i] - target[i];
    loss_terms[i] = d * d;
    grad[i] = d * grad_scale;
}

__kernel void predict_classes(
    __global const float *probs,
    __global int *out,
    const int N,
    const int C)
{
    int r = get_global_id(0);
    if (r >= N) return;
    int base = r * C;
    float maxv = probs[base];
    int maxj = 0;
    for (int c = 1; c < C; ++c) {
        float v = probs[base + c];
        if (v > maxv) {
            maxv = v;
            maxj = c;
        }
    }
    out[r] = maxj + 1;
}

__kernel void sgd_update(
    __global float *W,
    __global const float *dW,
    const float lr,
    const int total)
{
    int i = get_global_id(0);
    if (i >= total) return;
    W[i] -= lr * dW[i];
}

__kernel void momentum_update(
    __global float *W,
    __global const float *dW,
    __global float *V,
    const float lr,
    const float mu,
    const int total)
{
    int i = get_global_id(0);
    if (i >= total) return;
    float v = mu * V[i] - lr * dW[i];
    V[i] = v;
    W[i] += v;
}

__kernel void adam_update(
    __global float *W,
    __global const float *dW,
    __global float *m,
    __global float *v,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float beta1t,
    const float beta2t,
    const int total)
{
    int i = get_global_id(0);
    if (i >= total) return;
    float g = dW[i];
    float mt = beta1 * m[i] + (1.0f - beta1) * g;
    float vt = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = mt;
    v[i] = vt;
    float mhat = mt / (1.0f - beta1t);
    float vhat = vt / (1.0f - beta2t);
    W[i] -= lr * (mhat / (sqrt(vhat) + eps));
}
]=]

local function init_ffi_backend()
    if not jit then
        return nil, plain_lua_gpu_reason("The vendored native bridge was not loaded for this runtime.")
    end

    local ok, ffi = pcall(require, "ffi")
    if not ok then
        return nil, "LuaJIT FFI is unavailable: " .. tostring(ffi)
    end

    local is_windows = platform_separator() == "\\"
    local call = is_windows and "__stdcall" or ""
    local cdef = ([[
typedef signed char cl_char;
typedef unsigned char cl_uchar;
typedef short cl_short;
typedef unsigned short cl_ushort;
typedef int cl_int;
typedef unsigned int cl_uint;
typedef long long cl_long;
typedef unsigned long long cl_ulong;
typedef cl_ulong cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_mem_flags;
typedef cl_bitfield cl_command_queue_properties;
typedef unsigned int cl_bool;
typedef intptr_t cl_context_properties;

typedef struct _cl_platform_id *cl_platform_id;
typedef struct _cl_device_id *cl_device_id;
typedef struct _cl_context *cl_context;
typedef struct _cl_command_queue *cl_command_queue;
typedef struct _cl_mem *cl_mem;
typedef struct _cl_program *cl_program;
typedef struct _cl_kernel *cl_kernel;
typedef struct _cl_event *cl_event;

cl_int CL_CALL clGetPlatformIDs(cl_uint, cl_platform_id *, cl_uint *);
cl_int CL_CALL clGetPlatformInfo(cl_platform_id, cl_uint, size_t, void *, size_t *);
cl_int CL_CALL clGetDeviceIDs(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
cl_int CL_CALL clGetDeviceInfo(cl_device_id, cl_uint, size_t, void *, size_t *);
cl_context CL_CALL clCreateContext(const cl_context_properties *, cl_uint, const cl_device_id *, void *, void *, cl_int *);
cl_command_queue CL_CALL clCreateCommandQueue(cl_context, cl_device_id, cl_command_queue_properties, cl_int *);
cl_mem CL_CALL clCreateBuffer(cl_context, cl_mem_flags, size_t, void *, cl_int *);
cl_program CL_CALL clCreateProgramWithSource(cl_context, cl_uint, const char **, const size_t *, cl_int *);
cl_int CL_CALL clBuildProgram(cl_program, cl_uint, const cl_device_id *, const char *, void *, void *);
cl_int CL_CALL clGetProgramBuildInfo(cl_program, cl_device_id, cl_uint, size_t, void *, size_t *);
cl_kernel CL_CALL clCreateKernel(cl_program, const char *, cl_int *);
cl_int CL_CALL clSetKernelArg(cl_kernel, cl_uint, size_t, const void *);
cl_int CL_CALL clEnqueueWriteBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *, cl_uint, const cl_event *, cl_event *);
cl_int CL_CALL clEnqueueReadBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
cl_int CL_CALL clEnqueueNDRangeKernel(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
cl_int CL_CALL clFinish(cl_command_queue);
cl_int CL_CALL clReleaseMemObject(cl_mem);
cl_int CL_CALL clReleaseKernel(cl_kernel);
cl_int CL_CALL clReleaseProgram(cl_program);
cl_int CL_CALL clReleaseCommandQueue(cl_command_queue);
cl_int CL_CALL clReleaseContext(cl_context);
]]):gsub("CL_CALL", call)

    local ok_cdef, cdef_err = pcall(ffi.cdef, cdef)
    if not ok_cdef then
        local msg = tostring(cdef_err)
        if not msg:match("redefinition") then
            return nil, "OpenCL FFI declarations failed: " .. msg
        end
    end

    local lib
    local tried = {}
    local candidates = is_windows and { "OpenCL" } or { "OpenCL", "libOpenCL.so.1", "libOpenCL.so" }
    for _,name in ipairs(candidates) do
        local ok_load, loaded = pcall(ffi.load, name)
        if ok_load then
            lib = loaded
            break
        end
        tried[#tried+1] = name .. ": " .. tostring(loaded)
    end
    if not lib then
        return nil, "OpenCL runtime not found (" .. table.concat(tried, "; ") .. ")"
    end

    local CL_SUCCESS = 0
    local CL_TRUE = 1
    local CL_DEVICE_TYPE_GPU = 4
    local CL_MEM_READ_WRITE = 1
    local CL_PLATFORM_NAME = 0x0902
    local CL_PLATFORM_VENDOR = 0x0903
    local CL_DEVICE_NAME = 0x102B
    local CL_DEVICE_VENDOR = 0x102C
    local CL_DRIVER_VERSION = 0x102D
    local CL_DEVICE_VERSION = 0x102F
    local CL_PROGRAM_BUILD_LOG = 0x1183

    local function check(code, where)
        if code ~= CL_SUCCESS then
            error(where .. " failed with OpenCL error " .. tonumber(code), 2)
        end
    end

    local function info_string(kind, id, param)
        local sizep = ffi.new("size_t[1]")
        local code
        if kind == "platform" then
            code = lib.clGetPlatformInfo(id, param, 0, nil, sizep)
        else
            code = lib.clGetDeviceInfo(id, param, 0, nil, sizep)
        end
        if code ~= CL_SUCCESS or tonumber(sizep[0]) <= 0 then return "" end
        local buf = ffi.new("char[?]", tonumber(sizep[0]))
        if kind == "platform" then
            code = lib.clGetPlatformInfo(id, param, sizep[0], buf, nil)
        else
            code = lib.clGetDeviceInfo(id, param, sizep[0], buf, nil)
        end
        if code ~= CL_SUCCESS then return "" end
        return ffi.string(buf)
    end

    local num_platforms = ffi.new("cl_uint[1]")
    check(lib.clGetPlatformIDs(0, nil, num_platforms), "clGetPlatformIDs(count)")
    if tonumber(num_platforms[0]) == 0 then
        return nil, "no OpenCL platforms found"
    end

    local platforms = ffi.new("cl_platform_id[?]", tonumber(num_platforms[0]))
    check(lib.clGetPlatformIDs(num_platforms[0], platforms, nil), "clGetPlatformIDs(list)")

    local platform, device
    for i=0,tonumber(num_platforms[0])-1 do
        local ndev = ffi.new("cl_uint[1]")
        local code = lib.clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nil, ndev)
        if code == CL_SUCCESS and tonumber(ndev[0]) > 0 then
            local devices = ffi.new("cl_device_id[?]", tonumber(ndev[0]))
            check(lib.clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, ndev[0], devices, nil), "clGetDeviceIDs(list)")
            platform = platforms[i]
            device = devices[0]
            break
        end
    end

    if platform == nil or device == nil then
        return nil, "no OpenCL GPU device found"
    end

    local errp = ffi.new("cl_int[1]")
    local devices_one = ffi.new("cl_device_id[1]", { device })
    local context = lib.clCreateContext(nil, 1, devices_one, nil, nil, errp)
    check(errp[0], "clCreateContext")
    context = ffi.gc(context, lib.clReleaseContext)

    local queue = lib.clCreateCommandQueue(context, device, 0, errp)
    check(errp[0], "clCreateCommandQueue")
    queue = ffi.gc(queue, lib.clReleaseCommandQueue)

    local src = ffi.new("const char *[1]", { KERNEL_SOURCE })
    local lens = ffi.new("size_t[1]", { #KERNEL_SOURCE })
    local program = lib.clCreateProgramWithSource(context, 1, src, lens, errp)
    check(errp[0], "clCreateProgramWithSource")
    program = ffi.gc(program, lib.clReleaseProgram)

    local build_code = lib.clBuildProgram(program, 1, devices_one, nil, nil, nil)
    if build_code ~= CL_SUCCESS then
        local log_size = ffi.new("size_t[1]")
        lib.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nil, log_size)
        local log = ffi.new("char[?]", math.max(1, tonumber(log_size[0])))
        lib.clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size[0], log, nil)
        return nil, "OpenCL kernel build failed: " .. ffi.string(log)
    end

    local kernel_names = {
        "gather_rows",
        "matmul_bias",
        "matmul_ta",
        "matmul_tb",
        "sum_rows",
        "relu_forward",
        "relu_backward",
        "sigmoid_forward",
        "sigmoid_backward",
        "tanh_forward",
        "tanh_backward",
        "softmax_ce",
        "softmax_ce_backward",
        "mse_loss_grad",
        "predict_classes",
        "sgd_update",
        "momentum_update",
        "adam_update",
    }

    local kernels = {}
    for _,name in ipairs(kernel_names) do
        local k = lib.clCreateKernel(program, name, errp)
        check(errp[0], "clCreateKernel(" .. name .. ")")
        kernels[name] = ffi.gc(k, lib.clReleaseKernel)
    end

    local ctx = {
        ffi = ffi,
        cl = lib,
        CL_TRUE = CL_TRUE,
        CL_MEM_READ_WRITE = CL_MEM_READ_WRITE,
        context = context,
        queue = queue,
        device = device,
        platform = platform,
        kernels = kernels,
        info = {
            backend = "luajit-ffi-opencl",
            platform = info_string("platform", platform, CL_PLATFORM_NAME),
            platform_vendor = info_string("platform", platform, CL_PLATFORM_VENDOR),
            device = info_string("device", device, CL_DEVICE_NAME),
            device_vendor = info_string("device", device, CL_DEVICE_VENDOR),
            driver = info_string("device", device, CL_DRIVER_VERSION),
            opencl = info_string("device", device, CL_DEVICE_VERSION),
        },
    }

    function ctx:check(code, where)
        check(code, where)
    end

    function ctx:new_buffer(count)
        local bytes = count * ffi.sizeof("float")
        local err = ffi.new("cl_int[1]")
        local buf = self.cl.clCreateBuffer(self.context, self.CL_MEM_READ_WRITE, bytes, nil, err)
        self:check(err[0], "clCreateBuffer")
        return ffi.gc(buf, self.cl.clReleaseMemObject)
    end

    function ctx:tensor(rows, cols)
        rows = rows or 1
        cols = cols or 1
        local count = rows * cols
        return {
            buffer = self:new_buffer(count),
            rows = rows,
            cols = cols,
            count = count,
        }
    end

    function ctx:tensor_from_matrix(A)
        local rows, cols = shape(A)
        local t = self:tensor(rows, cols)
        self:write_matrix(t, A)
        return t
    end

    function ctx:write_matrix(t, A)
        local rows, cols = shape(A)
        assert(rows == t.rows and cols == t.cols, "GPU tensor shape mismatch")
        local arr = ffi.new("float[?]", t.count)
        local p = 0
        for i=1,rows do
            local row = A[i]
            for j=1,cols do
                arr[p] = row[j]
                p = p + 1
            end
        end
        self:check(self.cl.clEnqueueWriteBuffer(self.queue, t.buffer, self.CL_TRUE, 0, t.count * ffi.sizeof("float"), arr, 0, nil, nil), "clEnqueueWriteBuffer(matrix)")
    end

    function ctx:read_matrix(t)
        local arr = ffi.new("float[?]", t.count)
        self:check(self.cl.clEnqueueReadBuffer(self.queue, t.buffer, self.CL_TRUE, 0, t.count * ffi.sizeof("float"), arr, 0, nil, nil), "clEnqueueReadBuffer(matrix)")
        local out = {}
        local p = 0
        for i=1,t.rows do
            local row = {}
            for j=1,t.cols do
                row[j] = tonumber(arr[p])
                p = p + 1
            end
            out[i] = row
        end
        return out
    end

    function ctx:read_vector(t)
        local arr = ffi.new("float[?]", t.count)
        self:check(self.cl.clEnqueueReadBuffer(self.queue, t.buffer, self.CL_TRUE, 0, t.count * ffi.sizeof("float"), arr, 0, nil, nil), "clEnqueueReadBuffer(vector)")
        local out = {}
        for i=0,t.count-1 do out[i+1] = tonumber(arr[i]) end
        return out
    end

    function ctx:index_buffer(indices)
        local count = #indices
        local bytes = count * ffi.sizeof("int")
        local err = ffi.new("cl_int[1]")
        local buf = self.cl.clCreateBuffer(self.context, self.CL_MEM_READ_WRITE, bytes, nil, err)
        self:check(err[0], "clCreateBuffer(indices)")
        buf = ffi.gc(buf, self.cl.clReleaseMemObject)
        local arr = ffi.new("int[?]", count)
        for i=1,count do arr[i-1] = indices[i] end
        self:check(self.cl.clEnqueueWriteBuffer(self.queue, buf, self.CL_TRUE, 0, bytes, arr, 0, nil, nil), "clEnqueueWriteBuffer(indices)")
        return { buffer = buf, rows = count, cols = 1, count = count, is_int = true }
    end

    function ctx:set_arg(kernel, index, kind, value)
        local ref
        if kind == "mem" then
            ref = ffi.new("cl_mem[1]", { value.buffer })
            self:check(self.cl.clSetKernelArg(kernel, index, ffi.sizeof("cl_mem"), ref), "clSetKernelArg(mem)")
        elseif kind == "int" then
            ref = ffi.new("int[1]", value)
            self:check(self.cl.clSetKernelArg(kernel, index, ffi.sizeof("int"), ref), "clSetKernelArg(int)")
        elseif kind == "float" then
            ref = ffi.new("float[1]", value)
            self:check(self.cl.clSetKernelArg(kernel, index, ffi.sizeof("float"), ref), "clSetKernelArg(float)")
        else
            error("unknown OpenCL kernel arg kind: " .. tostring(kind))
        end
        return ref
    end

    function ctx:run_1d(name, n, args)
        local kernel = self.kernels[name]
        local refs = {}
        for i,arg in ipairs(args) do
            refs[i] = self:set_arg(kernel, i-1, arg[1], arg[2])
        end
        local global = ffi.new("size_t[1]", { math.max(1, n) })
        self:check(self.cl.clEnqueueNDRangeKernel(self.queue, kernel, 1, nil, global, nil, 0, nil, nil), "clEnqueueNDRangeKernel(" .. name .. ")")
        return refs
    end

    function ctx:run_2d(name, x, y, args)
        local kernel = self.kernels[name]
        local refs = {}
        for i,arg in ipairs(args) do
            refs[i] = self:set_arg(kernel, i-1, arg[1], arg[2])
        end
        local global = ffi.new("size_t[2]", { math.max(1, x), math.max(1, y) })
        self:check(self.cl.clEnqueueNDRangeKernel(self.queue, kernel, 2, nil, global, nil, 0, nil, nil), "clEnqueueNDRangeKernel(" .. name .. ")")
        return refs
    end

    function ctx:gather_rows(src, indices)
        local idx = self:index_buffer(indices)
        local out = self:tensor(#indices, src.cols)
        self:run_2d("gather_rows", #indices, src.cols, {
            { "mem", src },
            { "mem", idx },
            { "mem", out },
            { "int", #indices },
            { "int", src.cols },
        })
        return out
    end

    function ctx:linear_forward(x, W, b)
        assert(x.cols == W.rows, "linear forward dimension mismatch")
        local out = self:tensor(x.rows, W.cols)
        self:run_2d("matmul_bias", x.rows, W.cols, {
            { "mem", x },
            { "mem", W },
            { "mem", b },
            { "mem", out },
            { "int", x.rows },
            { "int", x.cols },
            { "int", W.cols },
        })
        return out
    end

    function ctx:matmul_ta(A, B)
        assert(A.rows == B.rows, "matmul_ta dimension mismatch")
        local out = self:tensor(A.cols, B.cols)
        self:run_2d("matmul_ta", A.cols, B.cols, {
            { "mem", A },
            { "mem", B },
            { "mem", out },
            { "int", A.rows },
            { "int", A.cols },
            { "int", B.cols },
        })
        return out
    end

    function ctx:matmul_tb(A, B)
        assert(A.cols == B.cols, "matmul_tb dimension mismatch")
        local out = self:tensor(A.rows, B.rows)
        self:run_2d("matmul_tb", A.rows, B.rows, {
            { "mem", A },
            { "mem", B },
            { "mem", out },
            { "int", A.rows },
            { "int", A.cols },
            { "int", B.rows },
        })
        return out
    end

    function ctx:sum_rows(A)
        local out = self:tensor(1, A.cols)
        self:run_1d("sum_rows", A.cols, {
            { "mem", A },
            { "mem", out },
            { "int", A.rows },
            { "int", A.cols },
        })
        return out
    end

    function ctx:activation_forward(kind, x)
        local out = self:tensor(x.rows, x.cols)
        if kind == "ReLU" then
            local mask = self:tensor(x.rows, x.cols)
            self:run_1d("relu_forward", x.count, {
                { "mem", x },
                { "mem", out },
                { "mem", mask },
                { "int", x.count },
            })
            return out, mask
        elseif kind == "Sigmoid" then
            self:run_1d("sigmoid_forward", x.count, {
                { "mem", x },
                { "mem", out },
                { "int", x.count },
            })
            return out
        elseif kind == "Tanh" then
            self:run_1d("tanh_forward", x.count, {
                { "mem", x },
                { "mem", out },
                { "int", x.count },
            })
            return out
        end
        error("unsupported activation: " .. tostring(kind))
    end

    function ctx:activation_backward(kind, dout, cache)
        local dx = self:tensor(dout.rows, dout.cols)
        if kind == "ReLU" then
            self:run_1d("relu_backward", dout.count, {
                { "mem", dout },
                { "mem", cache.mask },
                { "mem", dx },
                { "int", dout.count },
            })
        elseif kind == "Sigmoid" then
            self:run_1d("sigmoid_backward", dout.count, {
                { "mem", dout },
                { "mem", cache.out },
                { "mem", dx },
                { "int", dout.count },
            })
        elseif kind == "Tanh" then
            self:run_1d("tanh_backward", dout.count, {
                { "mem", dout },
                { "mem", cache.out },
                { "mem", dx },
                { "int", dout.count },
            })
        else
            error("unsupported activation: " .. tostring(kind))
        end
        return dx
    end

    function ctx:softmax_ce(logits, target)
        assert(logits.rows == target.rows and logits.cols == target.cols, "softmax target shape mismatch")
        local probs = self:tensor(logits.rows, logits.cols)
        local losses = self:tensor(logits.rows, 1)
        self:run_1d("softmax_ce", logits.rows, {
            { "mem", logits },
            { "mem", target },
            { "mem", probs },
            { "mem", losses },
            { "int", logits.rows },
            { "int", logits.cols },
        })
        local loss_terms = self:read_vector(losses)
        local loss = 0.0
        for i=1,#loss_terms do loss = loss + loss_terms[i] end
        return loss / logits.rows, probs
    end

    function ctx:softmax_ce_backward(probs, target)
        local out = self:tensor(probs.rows, probs.cols)
        self:run_1d("softmax_ce_backward", probs.count, {
            { "mem", probs },
            { "mem", target },
            { "mem", out },
            { "float", 1.0 / probs.rows },
            { "int", probs.count },
        })
        return out
    end

    function ctx:mse_loss_grad(pred, target)
        assert(pred.rows == target.rows and pred.cols == target.cols, "MSE target shape mismatch")
        local grad = self:tensor(pred.rows, pred.cols)
        local loss_terms = self:tensor(pred.rows, pred.cols)
        self:run_1d("mse_loss_grad", pred.count, {
            { "mem", pred },
            { "mem", target },
            { "mem", grad },
            { "mem", loss_terms },
            { "float", 2.0 / pred.count },
            { "int", pred.count },
        })
        local terms = self:read_vector(loss_terms)
        local sse = 0.0
        for i=1,#terms do sse = sse + terms[i] end
        return sse / pred.count, grad
    end

    function ctx:predict_classes(probs)
        local err = ffi.new("cl_int[1]")
        local bytes = probs.rows * ffi.sizeof("int")
        local buf = self.cl.clCreateBuffer(self.context, self.CL_MEM_READ_WRITE, bytes, nil, err)
        self:check(err[0], "clCreateBuffer(predictions)")
        buf = ffi.gc(buf, self.cl.clReleaseMemObject)
        local pred = { buffer = buf, rows = probs.rows, cols = 1, count = probs.rows, is_int = true }
        self:run_1d("predict_classes", probs.rows, {
            { "mem", probs },
            { "mem", pred },
            { "int", probs.rows },
            { "int", probs.cols },
        })
        local arr = ffi.new("int[?]", probs.rows)
        self:check(self.cl.clEnqueueReadBuffer(self.queue, pred.buffer, self.CL_TRUE, 0, bytes, arr, 0, nil, nil), "clEnqueueReadBuffer(predictions)")
        local out = {}
        for i=0,probs.rows-1 do out[i+1] = tonumber(arr[i]) end
        return out
    end

    function ctx:update_param(algo, W, dW, m, v, opt, t)
        local lr = opt.lr or (algo == "adam" and 1e-3 or 1e-2)
        if algo == "sgd" then
            self:run_1d("sgd_update", W.count, {
                { "mem", W },
                { "mem", dW },
                { "float", lr },
                { "int", W.count },
            })
        elseif algo == "momentum" then
            self:run_1d("momentum_update", W.count, {
                { "mem", W },
                { "mem", dW },
                { "mem", m },
                { "float", lr },
                { "float", opt.mu or 0.9 },
                { "int", W.count },
            })
        else
            local beta1 = opt.beta1 or 0.9
            local beta2 = opt.beta2 or 0.999
            self:run_1d("adam_update", W.count, {
                { "mem", W },
                { "mem", dW },
                { "mem", m },
                { "mem", v },
                { "float", lr },
                { "float", beta1 },
                { "float", beta2 },
                { "float", opt.eps or 1e-8 },
                { "float", beta1 ^ t },
                { "float", beta2 ^ t },
                { "int", W.count },
            })
        end
    end

    return ctx
end

------------------------------------------------------------
-- Backend lifecycle
------------------------------------------------------------

local function ensure_context()
    if GPU.ctx then return GPU.ctx end
    if GPU.init_attempted then return nil, GPU.unavailable_reason end
    GPU.init_attempted = true

    local bridge, bridge_err = try_load_native_bridge()
    if bridge and bridge.kind == "opencl" then
        GPU.native_bridge = bridge
        if type(bridge.probe) == "function" then
            local ok_probe, probe = pcall(bridge.probe)
            if ok_probe and type(probe) == "table" then
                GPU.native_probe = probe
            end
        end
        if bridge.not_yet_supported == true and not jit then
            local probe_suffix = ""
            if GPU.native_probe and GPU.native_probe.available then
                probe_suffix = "The native bridge loaded and found OpenCL device "
                    .. tostring(GPU.native_probe.device or "unknown")
                    .. ", but it only exposes probing today."
            else
                probe_suffix = "The native bridge loaded, but it only exposes probing today."
            end
            GPU.unavailable_reason = plain_lua_gpu_reason(probe_suffix)
            return nil, GPU.unavailable_reason
        elseif bridge.not_yet_supported ~= true then
            GPU.unavailable_reason = "vendored native bridge is present but this Lua wrapper only supports LuaJIT FFI execution in this version"
            return nil, GPU.unavailable_reason
        end
    end

    local ctx, ffi_err = init_ffi_backend()
    if ctx then
        ctx.native_bridge_error = bridge_err
        GPU.ctx = ctx
        return ctx
    end

    GPU.unavailable_reason = ffi_err or bridge_err or "OpenCL backend unavailable"
    return nil, GPU.unavailable_reason
end

local function resolve_device(self, opts)
    opts = opts or {}
    local requested = opts.device or self.device or "cpu"
    requested = tostring(requested):lower()
    if requested == "gpu" or requested == "opencl" then
        return "gpu"
    elseif requested == "auto" then
        local ctx = ensure_context()
        return ctx and "gpu" or "cpu"
    end
    return "cpu"
end

local function should_use_gpu(self, opts)
    opts = opts or {}
    local device = resolve_device(self, opts)
    if device ~= "gpu" then return false end

    local ctx, reason = ensure_context()
    if ctx then return true, ctx end

    if opts.gpu_required or self.gpu_required then
        error("GPU backend requested but unavailable: " .. tostring(reason), 3)
    end
    warn_once("GPU backend unavailable (" .. tostring(reason) .. "); falling back to CPU")
    return false
end

------------------------------------------------------------
-- Network execution on GPU
------------------------------------------------------------

local function upload_linear(ctx, L)
    if L._gpu and L._gpu.ctx == ctx then return L._gpu end
    L._gpu = {
        ctx = ctx,
        W = ctx:tensor_from_matrix(L.W),
        b = ctx:tensor_from_matrix(L.b),
        dW = ctx:tensor_from_matrix(L.dW or M.zeros(L.in_dim, L.out_dim)),
        db = ctx:tensor_from_matrix(L.db or M.zeros(1, L.out_dim)),
        mW = ctx:tensor_from_matrix(L.mW or M.zeros(L.in_dim, L.out_dim)),
        vW = ctx:tensor_from_matrix(L.vW or M.zeros(L.in_dim, L.out_dim)),
        mB = ctx:tensor_from_matrix(L.mB or M.zeros(1, L.out_dim)),
        vB = ctx:tensor_from_matrix(L.vB or M.zeros(1, L.out_dim)),
    }
    return L._gpu
end

local function sync_linear_from_gpu(ctx, L)
    if not L._gpu or L._gpu.ctx ~= ctx then return end
    L.W = ctx:read_matrix(L._gpu.W)
    L.b = ctx:read_matrix(L._gpu.b)
    L.dW = ctx:read_matrix(L._gpu.dW)
    L.db = ctx:read_matrix(L._gpu.db)
    L.mW = ctx:read_matrix(L._gpu.mW)
    L.vW = ctx:read_matrix(L._gpu.vW)
    L.mB = ctx:read_matrix(L._gpu.mB)
    L.vB = ctx:read_matrix(L._gpu.vB)
end

local function sync_net_from_gpu(ctx, net)
    for _,L in ipairs(net.layers) do
        if L.type == "Linear" then
            sync_linear_from_gpu(ctx, L)
        end
    end
end

local function gpu_forward_tensor(ctx, net, x)
    local out = x
    for _,L in ipairs(net.layers) do
        if L.type == "Linear" then
            local G = upload_linear(ctx, L)
            L._gpu_cache = { x = out }
            out = ctx:linear_forward(out, G.W, G.b)
            L._gpu_cache.out = out
        elseif L.type == "ReLU" or L.type == "Sigmoid" or L.type == "Tanh" then
            local y, mask = ctx:activation_forward(L.type, out)
            L._gpu_cache = { x = out, out = y, mask = mask }
            out = y
        else
            error("unsupported GPU layer type: " .. tostring(L.type))
        end
    end
    return out
end

local function gpu_backward_tensor(ctx, net, grad)
    local dout = grad
    for i=#net.layers,1,-1 do
        local L = net.layers[i]
        if L.type == "Linear" then
            local G = upload_linear(ctx, L)
            local cache = assert(L._gpu_cache, "missing GPU forward cache")
            G.dW = ctx:matmul_ta(cache.x, dout)
            G.db = ctx:sum_rows(dout)
            L._gpu.dW = G.dW
            L._gpu.db = G.db
            dout = ctx:matmul_tb(dout, G.W)
        elseif L.type == "ReLU" or L.type == "Sigmoid" or L.type == "Tanh" then
            dout = ctx:activation_backward(L.type, dout, assert(L._gpu_cache, "missing GPU activation cache"))
        else
            error("unsupported GPU layer type: " .. tostring(L.type))
        end
    end
    return dout
end

local function gpu_step(ctx, net, opt)
    opt = opt or {}
    local algo = (opt.algo or "adam"):lower()
    for _,L in ipairs(net.layers) do
        if L.type == "Linear" then
            local G = upload_linear(ctx, L)
            L.t = (L.t or 0) + 1
            ctx:update_param(algo, G.W, G.dW, G.mW, G.vW, opt, L.t)
            ctx:update_param(algo, G.b, G.db, G.mB, G.vB, opt, L.t)
        end
    end
end

local function gpu_predict_proba_tensor(ctx, net, X_tensor)
    local logits = gpu_forward_tensor(ctx, net, X_tensor)
    local dummy_target = ctx:tensor(logits.rows, logits.cols)
    local zeros = M.zeros(logits.rows, logits.cols)
    ctx:write_matrix(dummy_target, zeros)
    local _, probs = ctx:softmax_ce(logits, dummy_target)
    return probs
end

local function gpu_forward_to_table(ctx, net, X)
    local x = ctx:tensor_from_matrix(X)
    local out = gpu_forward_tensor(ctx, net, x)
    return ctx:read_matrix(out)
end

local function gpu_predict_proba_to_table(ctx, net, X)
    local x = ctx:tensor_from_matrix(X)
    local probs = gpu_predict_proba_tensor(ctx, net, x)
    return ctx:read_matrix(probs)
end

local function gpu_predict_classes_to_table(ctx, net, X)
    local x = ctx:tensor_from_matrix(X)
    local probs = gpu_predict_proba_tensor(ctx, net, x)
    return ctx:predict_classes(probs)
end

local function gpu_fit(ctx, net, X, y, cfg)
    cfg = cfg or {}
    local epochs = cfg.epochs or 50
    local batch = cfg.batch or 16
    local lr = cfg.lr or 1e-3
    local algo = cfg.algo or "adam"
    local verbose = (cfg.verbose ~= false)
    local task = cfg.task or "auto"

    local N = select(1, shape(X))
    local X_gpu = ctx:tensor_from_matrix(X)
    local target_matrix, criterion_type

    if task == "regression" then
        target_matrix = y
        criterion_type = "mse"
    else
        target_matrix = select(1, ensure_targets(y, cfg.num_classes))
        criterion_type = "softmax"
    end

    local Y_gpu = ctx:tensor_from_matrix(target_matrix)
    local num_batches = math.max(1, math.floor((N + batch - 1) / batch))

    for epoch=1,epochs do
        local idx = shuffle_indices(N)
        local epoch_loss = 0.0

        for b=1,num_batches do
            local s = (b-1) * batch + 1
            local e = math.min(b * batch, N)
            if s > e then break end

            local indices = make_batch_indices(idx, s, e)
            local xbatch = ctx:gather_rows(X_gpu, indices)
            local ybatch = ctx:gather_rows(Y_gpu, indices)
            local pred = gpu_forward_tensor(ctx, net, xbatch)
            local loss, dloss

            if criterion_type == "softmax" then
                local probs
                loss, probs = ctx:softmax_ce(pred, ybatch)
                dloss = ctx:softmax_ce_backward(probs, ybatch)
            else
                loss, dloss = ctx:mse_loss_grad(pred, ybatch)
            end

            epoch_loss = epoch_loss + loss
            gpu_backward_tensor(ctx, net, dloss)
            gpu_step(ctx, net, {
                algo = algo,
                lr = lr,
                beta1 = cfg.beta1,
                beta2 = cfg.beta2,
                eps = cfg.eps,
                mu = cfg.mu,
            })
        end

        epoch_loss = epoch_loss / num_batches

        if verbose then
            if criterion_type == "softmax" then
                local preds = ctx:predict_classes(gpu_predict_proba_tensor(ctx, net, X_gpu))
                local acc = NN.metrics.accuracy_from_indices(preds, y)
                print(("epoch %d | loss %.6f | acc %.2f%%"):format(epoch, epoch_loss, 100 * acc))
            else
                print(("epoch %d | loss %.6f"):format(epoch, epoch_loss))
            end
        end
    end

    sync_net_from_gpu(ctx, net)
end

------------------------------------------------------------
-- Public API patching
------------------------------------------------------------

local function patch_net_metatable()
    local sample = NN.new()
    local Net = getmetatable(sample)
    if Net.__luann_gpu_patched then return end
    Net.__luann_gpu_patched = true

    original.forward = Net.forward
    original.fit = Net.fit
    original.step = Net.step
    original.predict_proba = Net.predict_proba
    original.predict_classes = Net.predict_classes

    function Net:forward(X, opts)
        local use_gpu, ctx = should_use_gpu(self, opts)
        if use_gpu then
            return gpu_forward_to_table(ctx, self, X)
        end
        return original.forward(self, X)
    end

    function Net:fit(X, y, cfg)
        cfg = cfg or {}
        local use_gpu, ctx = should_use_gpu(self, cfg)
        if use_gpu then
            return gpu_fit(ctx, self, X, y, cfg)
        end
        local cpu_cfg = copy_table(cfg)
        cpu_cfg.device = nil
        cpu_cfg.gpu_required = nil
        local result = original.fit(self, X, y, cpu_cfg)
        clear_gpu_state(self)
        return result
    end

    function Net:step(opt)
        local result = original.step(self, opt)
        clear_gpu_state(self)
        return result
    end

    function Net:predict_proba(X, opts)
        local use_gpu, ctx = should_use_gpu(self, opts)
        if use_gpu then
            return gpu_predict_proba_to_table(ctx, self, X)
        end
        return original.predict_proba(self, X)
    end

    function Net:predict_classes(X, opts)
        local use_gpu, ctx = should_use_gpu(self, opts)
        if use_gpu then
            return gpu_predict_classes_to_table(ctx, self, X)
        end
        return original.predict_classes(self, X)
    end
end

local original_new = NN.new
local original_load_state = NN.load_state
patch_net_metatable()

NN.new = function(opts)
    local net = original_new()
    if type(opts) == "table" then
        net.device = opts.device
        net.gpu_required = opts.gpu_required
    end
    return net
end

NN.load_state = function(net, state)
    local result = original_load_state(net, state)
    clear_gpu_state(net)
    return result
end

function NN.gpu_available()
    return ensure_context() ~= nil
end

function NN.gpu_info()
    local ctx, reason = ensure_context()
    if ctx then
        return copy_table(ctx.info)
    end
    return {
        available = false,
        reason = reason,
        expected_bridge = native_bridge_path(),
        native_bridge_loaded = GPU.native_bridge ~= nil,
        native_probe = GPU.native_probe,
        runtime = jit and "LuaJIT" or _VERSION,
    }
end

function NN.available_devices()
    local devices = {
        { device = "cpu", available = true, backend = "lua-table" },
    }
    local ctx = ensure_context()
    if ctx then
        devices[#devices+1] = {
            device = "gpu",
            available = true,
            backend = ctx.info.backend,
            name = ctx.info.device,
            vendor = ctx.info.device_vendor,
        }
    else
        devices[#devices+1] = {
            device = "gpu",
            available = false,
            reason = GPU.unavailable_reason,
            expected_bridge = native_bridge_path(),
            native_bridge_loaded = GPU.native_bridge ~= nil,
        }
    end
    return devices
end

NN.GPU = {
    available = NN.gpu_available,
    info = NN.gpu_info,
    available_devices = NN.available_devices,
    native_bridge_path = native_bridge_path,
    kernel_source = function() return KERNEL_SOURCE end,
}

return NN

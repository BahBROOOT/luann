# luann OpenCL Native Assets

This directory contains the optional native bridge for plain Lua runtimes.

`nn_gpu_backend.lua` currently has the complete executable OpenCL path for
LuaJIT through FFI. That LuaJIT path dynamically loads `OpenCL.dll` on Windows
or `libOpenCL.so.1`/`libOpenCL.so` on Linux and embeds its kernels as a Lua
string, so it needs no OpenCL SDK, Lua package, or external kernel file.

The native bridge follows the same self-contained rule:

- no OpenCL headers
- no OpenCL import library
- no Lua headers
- no Lua import library
- runtime dynamic loading of the system OpenCL driver
- runtime lookup of the Lua C API from the already-loaded Lua DLL/shared object
- ABI-specific binaries placed under `native/bin/`

## Build Targets

Expected binary layout:

- `native/bin/windows-x64/lua54/luann_opencl.dll`
- `native/bin/windows-x64/luajit/luann_opencl.dll`
- `native/bin/linux-x64/lua54/luann_opencl.so`
- `native/bin/linux-x64/luajit/luann_opencl.so`

## Windows

```powershell
.\native\build_windows.ps1 -LuaAbi lua54
.\native\build_windows.ps1 -LuaAbi luajit
```

The script uses `cl` if it is already on `PATH`; otherwise it tries to locate
Visual Studio C++ tools with `vswhere.exe`.

## Linux

```sh
LUA_ABI=lua54 \
sh native/build_linux.sh

LUA_ABI=luajit \
sh native/build_linux.sh
```

The bridge currently exposes a runtime probe and ABI packaging target. Full
tensor execution for plain Lua should be added to `opencl_bridge.c`; LuaJIT
already runs the full backend directly from `nn_gpu_backend.lua`.

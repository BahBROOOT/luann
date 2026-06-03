Prebuilt OpenCL bridge binaries belong in this directory.

Expected layout:

- `windows-x64/lua54/luann_opencl.dll`
- `windows-x64/luajit/luann_opencl.dll`
- `linux-x64/lua54/luann_opencl.so`
- `linux-x64/luajit/luann_opencl.so`

The LuaJIT backend in `nn_gpu_backend.lua` can execute directly through FFI
without these files. Plain Lua 5.x needs the matching binary for its Lua ABI.

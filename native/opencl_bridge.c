/*
 * luann_opencl native bridge bootstrap.
 *
 * This file intentionally includes no OpenCL headers and links no OpenCL
 * import library. It loads OpenCL.dll/libOpenCL.so.1 at runtime so release
 * binaries can stay self-contained apart from the user's GPU driver runtime.
 *
 * The current Lua implementation uses LuaJIT FFI for full tensor execution.
 * This bridge provides the repo-contained plain-Lua ABI target and a runtime
 * probe; the tensor entrypoints should be added here when building Lua 5.x
 * binaries.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

typedef struct lua_State lua_State;
typedef int (*lua_CFunction)(lua_State *L);
typedef long long lua_Integer;

typedef struct LuannLua {
    void (*createtable)(lua_State *L, int narr, int nrec);
    void (*pushboolean)(lua_State *L, int b);
    const char *(*pushstring)(lua_State *L, const char *s);
    void (*pushinteger)(lua_State *L, lua_Integer n);
    void (*pushcclosure)(lua_State *L, lua_CFunction fn, int n);
    void (*setfield)(lua_State *L, int idx, const char *k);
} LuannLua;

static LuannLua LUA;

#if defined(_WIN32)
#define LUANN_EXPORT __declspec(dllexport)
#else
#define LUANN_EXPORT __attribute__((visibility("default")))
#endif

static void *load_lua_symbol(const char *name) {
#if defined(_WIN32)
    const char *modules[] = {
        "lua54.dll",
        "lua53.dll",
        "lua52.dll",
        "lua51.dll",
        "lua.dll",
        NULL
    };
    int i;
    for (i = 0; modules[i]; ++i) {
        HMODULE mod = GetModuleHandleA(modules[i]);
        if (mod) {
            void *sym = (void *)GetProcAddress(mod, name);
            if (sym) return sym;
        }
    }
    return NULL;
#else
    return dlsym(RTLD_DEFAULT, name);
#endif
}

static int load_lua_api(void) {
    LUA.createtable = (void (*)(lua_State *, int, int))load_lua_symbol("lua_createtable");
    LUA.pushboolean = (void (*)(lua_State *, int))load_lua_symbol("lua_pushboolean");
    LUA.pushstring = (const char *(*)(lua_State *, const char *))load_lua_symbol("lua_pushstring");
    LUA.pushinteger = (void (*)(lua_State *, lua_Integer))load_lua_symbol("lua_pushinteger");
    LUA.pushcclosure = (void (*)(lua_State *, lua_CFunction, int))load_lua_symbol("lua_pushcclosure");
    LUA.setfield = (void (*)(lua_State *, int, const char *))load_lua_symbol("lua_setfield");

    return LUA.createtable && LUA.pushboolean && LUA.pushstring &&
        LUA.pushinteger && LUA.pushcclosure && LUA.setfield;
}

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef cl_ulong cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef struct _cl_platform_id *cl_platform_id;
typedef struct _cl_device_id *cl_device_id;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_GPU 4
#define CL_PLATFORM_NAME 0x0902
#define CL_DEVICE_NAME 0x102B

typedef cl_int (*p_clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
typedef cl_int (*p_clGetPlatformInfo)(cl_platform_id, cl_uint, size_t, void *, size_t *);
typedef cl_int (*p_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
typedef cl_int (*p_clGetDeviceInfo)(cl_device_id, cl_uint, size_t, void *, size_t *);

typedef struct LuannOpenCL {
    void *lib;
    p_clGetPlatformIDs clGetPlatformIDs;
    p_clGetPlatformInfo clGetPlatformInfo;
    p_clGetDeviceIDs clGetDeviceIDs;
    p_clGetDeviceInfo clGetDeviceInfo;
    char reason[256];
} LuannOpenCL;

static void set_reason(LuannOpenCL *ocl, const char *reason) {
    strncpy(ocl->reason, reason, sizeof(ocl->reason) - 1);
    ocl->reason[sizeof(ocl->reason) - 1] = '\0';
}

static void *load_symbol(LuannOpenCL *ocl, const char *name) {
#if defined(_WIN32)
    return (void *)GetProcAddress((HMODULE)ocl->lib, name);
#else
    return dlsym(ocl->lib, name);
#endif
}

static int load_opencl(LuannOpenCL *ocl) {
    memset(ocl, 0, sizeof(*ocl));

#if defined(_WIN32)
    ocl->lib = (void *)LoadLibraryA("OpenCL.dll");
#else
    ocl->lib = dlopen("libOpenCL.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!ocl->lib) {
        ocl->lib = dlopen("libOpenCL.so", RTLD_LAZY | RTLD_LOCAL);
    }
#endif

    if (!ocl->lib) {
        set_reason(ocl, "OpenCL runtime library was not found");
        return 0;
    }

    ocl->clGetPlatformIDs = (p_clGetPlatformIDs)load_symbol(ocl, "clGetPlatformIDs");
    ocl->clGetPlatformInfo = (p_clGetPlatformInfo)load_symbol(ocl, "clGetPlatformInfo");
    ocl->clGetDeviceIDs = (p_clGetDeviceIDs)load_symbol(ocl, "clGetDeviceIDs");
    ocl->clGetDeviceInfo = (p_clGetDeviceInfo)load_symbol(ocl, "clGetDeviceInfo");

    if (!ocl->clGetPlatformIDs || !ocl->clGetPlatformInfo ||
        !ocl->clGetDeviceIDs || !ocl->clGetDeviceInfo) {
        set_reason(ocl, "OpenCL runtime is missing required symbols");
        return 0;
    }

    return 1;
}

static void unload_opencl(LuannOpenCL *ocl) {
    if (!ocl->lib) return;
#if defined(_WIN32)
    FreeLibrary((HMODULE)ocl->lib);
#else
    dlclose(ocl->lib);
#endif
    ocl->lib = NULL;
}

static void push_opencl_string(lua_State *L, LuannOpenCL *ocl, int platform, void *id, cl_uint param) {
    size_t n = 0;
    cl_int err;

    if (platform) {
        err = ocl->clGetPlatformInfo((cl_platform_id)id, param, 0, NULL, &n);
    } else {
        err = ocl->clGetDeviceInfo((cl_device_id)id, param, 0, NULL, &n);
    }
    if (err != CL_SUCCESS || n == 0) {
        LUA.pushstring(L, "");
        return;
    }

    char *buf = (char *)malloc(n);
    if (!buf) {
        LUA.pushstring(L, "");
        return;
    }

    if (platform) {
        err = ocl->clGetPlatformInfo((cl_platform_id)id, param, n, buf, NULL);
    } else {
        err = ocl->clGetDeviceInfo((cl_device_id)id, param, n, buf, NULL);
    }
    if (err != CL_SUCCESS) {
        free(buf);
        LUA.pushstring(L, "");
        return;
    }

    LUA.pushstring(L, buf);
    free(buf);
}

static int l_probe(lua_State *L) {
    LuannOpenCL ocl;
    cl_uint platform_count = 0;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_uint device_count = 0;

    LUA.createtable(L, 0, 0);

    if (!load_opencl(&ocl)) {
        LUA.pushboolean(L, 0);
        LUA.setfield(L, -2, "available");
        LUA.pushstring(L, ocl.reason);
        LUA.setfield(L, -2, "reason");
        unload_opencl(&ocl);
        return 1;
    }

    if (ocl.clGetPlatformIDs(0, NULL, &platform_count) != CL_SUCCESS || platform_count == 0) {
        LUA.pushboolean(L, 0);
        LUA.setfield(L, -2, "available");
        LUA.pushstring(L, "no OpenCL platforms found");
        LUA.setfield(L, -2, "reason");
        unload_opencl(&ocl);
        return 1;
    }

    if (ocl.clGetPlatformIDs(1, &platform, NULL) != CL_SUCCESS ||
        ocl.clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &device_count) != CL_SUCCESS ||
        device_count == 0) {
        LUA.pushboolean(L, 0);
        LUA.setfield(L, -2, "available");
        LUA.pushstring(L, "no OpenCL GPU device found");
        LUA.setfield(L, -2, "reason");
        unload_opencl(&ocl);
        return 1;
    }

    LUA.pushboolean(L, 1);
    LUA.setfield(L, -2, "available");
    LUA.pushinteger(L, (lua_Integer)platform_count);
    LUA.setfield(L, -2, "platform_count");
    push_opencl_string(L, &ocl, 1, platform, CL_PLATFORM_NAME);
    LUA.setfield(L, -2, "platform");
    push_opencl_string(L, &ocl, 0, device, CL_DEVICE_NAME);
    LUA.setfield(L, -2, "device");

    unload_opencl(&ocl);
    return 1;
}

LUANN_EXPORT int luaopen_luann_opencl(lua_State *L) {
    if (!load_lua_api()) {
        return 0;
    }

    LUA.createtable(L, 0, 0);

    LUA.pushstring(L, "opencl");
    LUA.setfield(L, -2, "kind");

    LUA.pushboolean(L, 1);
    LUA.setfield(L, -2, "not_yet_supported");

    LUA.pushcclosure(L, l_probe, 0);
    LUA.setfield(L, -2, "probe");

    LUA.pushstring(L, "native bridge probe compiled; tensor execution is currently provided by nn_gpu_backend.lua on LuaJIT");
    LUA.setfield(L, -2, "note");

    return 1;
}

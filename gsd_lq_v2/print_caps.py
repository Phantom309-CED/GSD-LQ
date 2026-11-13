import ctypes
import os

candidates = [
    os.path.expandvars("$PREFIX/opt/vendor/lib/libOpenCL.so"),
    os.path.expandvars("$PREFIX/lib/libOpenCL.so"),
    "/vendor/lib64/libOpenCL-pixel.so",
    "/vendor/lib64/libOpenCL.so",
    "/vendor/lib64/libPVROCL.so",
]

cl = None
for p in candidates:
    if os.path.exists(p):
        try:
            cl = ctypes.CDLL(p)
            print(f"[+] Loaded: {p}")
            break
        except Exception as e:
            pass

if not cl:
    print("[-] Could not load OpenCL driver")
    exit(1)

n_p = ctypes.c_uint(0)
cl.clGetPlatformIDs(0, None, ctypes.byref(n_p))
plats = (ctypes.c_void_p * n_p.value)()
cl.clGetPlatformIDs(n_p.value, plats, None)

for p in plats:
    pname = ctypes.create_string_buffer(256)
    cl.clGetPlatformInfo(p, 0x0902, 256, pname, None)
    print("Platform:", pname.value.decode(errors="replace"))
    
    n_d = ctypes.c_uint(0)
    cl.clGetDeviceIDs(p, 0xFFFFFFFF, 0, None, ctypes.byref(n_d))
    devs = (ctypes.c_void_p * n_d.value)()
    cl.clGetDeviceIDs(p, 0xFFFFFFFF, n_d.value, devs, None)
    for d in devs:
        def get_str(param):
            sz = ctypes.c_size_t(0)
            cl.clGetDeviceInfo(d, param, 0, None, ctypes.byref(sz))
            if sz.value == 0: return ""
            b = ctypes.create_string_buffer(sz.value)
            cl.clGetDeviceInfo(d, param, sz.value, b, None)
            return b.value.decode(errors="replace")
        
        def get_u32(param):
            v = ctypes.c_uint(0)
            cl.clGetDeviceInfo(d, param, ctypes.sizeof(v), ctypes.byref(v), None)
            return v.value

        print("  Device:", get_str(0x102B))
        print("  Version:", get_str(0x102F))
        print("  Driver Ver:", get_str(0x102C))
        print("  C Version:", get_str(0x103D))
        print("  IL Version (SPIR-V):", get_str(0x105B))
        print("  Compiler Available:", get_u32(0x1028))
        print("  Linker Available:", get_u32(0x103E))
        exts = get_str(0x102D).split()
        print(f"  Extensions ({len(exts)}):")
        for e in sorted(exts):
            print("   *", e)

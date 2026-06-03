------------------------------------------------------------
-- Canonical public entrypoint.
--
-- CPU is the default compute device. Pass { device = "gpu" } or
-- { device = "auto" } to training/inference calls to use the optional
-- OpenCL backend when available.
------------------------------------------------------------

return require("nn_gpu_backend")

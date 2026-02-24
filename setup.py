from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pathlib import Path

this_dir = Path(__file__).parent

setup(
    name="selective_scan",
    ext_modules=[
        CUDAExtension(
            name="selective_scan",
            sources=[
                str(this_dir / "csrc" / "selective_scan" / "selective_scan.cpp"),
                str(this_dir / "csrc" / "selective_scan" / "forward_kernel.cu"),
                str(this_dir / "csrc" / "selective_scan" / "backward_kernel.cu"),
            ],
            include_dirs=[
                str(this_dir / "csrc" / "selective_scan"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    "torch>=1.7",
    "torchvision>=0.8",
    "numpy>=1.20.0",
]

EXTRAS_REQUIRE = {
    "all": ["matplotlib", "pycocotools", "opencv-python", "onnx", "onnxruntime"],
    "dev": ["flake8", "isort", "black", "mypy"],
}

setup(
    name="segment_anything",
    version="1.0.0",
    description="Segment Anything Model (SAM) for promptable image segmentation.",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(exclude=("notebooks", "demo")),
    extras_require=EXTRAS_REQUIRE,
)

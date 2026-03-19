##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################

"""Test the ProfileModeImportGuard to ensure it works correctly."""

import sys

import pytest


def test_import_guard_blocks_non_stdlib():
    """Verify ProfileModeImportGuard blocks non-stdlib imports."""
    from conftest import ProfileModeImportGuard

    # Should allow stdlib imports
    with ProfileModeImportGuard():
        # Re-import to ensure guard doesn't block already-imported modules
        pass

    # Test blocking non-stdlib imports
    # Note: Must clear from sys.modules first since import hooks only
    # run for uncached modules
    test_packages = ["pandas", "yaml", "numpy"]

    for package in test_packages:
        # Clear from cache if present
        if package in sys.modules:
            del sys.modules[package]

        # Should block the import
        with pytest.raises(ImportError, match="PROFILE MODE DEPENDENCY VIOLATION"):
            with ProfileModeImportGuard():
                __import__(package)

        # Verify error message mentions the forbidden package
        if package in sys.modules:
            del sys.modules[package]
        with pytest.raises(ImportError, match=package):
            with ProfileModeImportGuard():
                __import__(package)

#!/usr/bin/env python3
"""Diagnostic script for inspecting VDevice parameter attributes and
HailoSchedulingAlgorithm in the installed hailo_platform 4.20.0 version.
This script does not create a VDevice and does not reserve the hardware; it
only prints the parameter structure and default values.
"""
import hailo_platform
from hailo_platform import VDevice

print("=== Scheduling-related attributes and enums in hailo_platform ===")
names = [n for n in dir(hailo_platform) if
         "sched" in n.lower() or "group" in n.lower() or "process" in n.lower()]
print(names)

print("\n=== VDevice.create_params() attributes ===")
params = VDevice.create_params()
attrs = [a for a in dir(params) if not a.startswith("_")]
print(attrs)

print("\n=== Current value of each attribute (default) ===")
for a in attrs:
    try:
        val = getattr(params, a)
        if not callable(val):
            print(f"{a} = {val!r}")
    except Exception as e:
        print(f"{a} -> read error: {e}")

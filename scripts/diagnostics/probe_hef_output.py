#!/usr/bin/env python3
"""
Diagnostic script (not part of the final runtime workflow). Run once to inspect
 the actual output structure of best.hef and avoid assumptions when implementing
 the official Stage 2 verification pipeline.

Usage from the project root:
    python3 scripts/diagnostics/probe_hef_output.py

After running, review or share the complete terminal output.
"""
import numpy as np
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, InferVStreams,
    ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType,
    HailoSchedulingAlgorithm,
)

HEF_PATH = "models/best.hef"  # Update this path if the model is stored elsewhere.


def main():
    hef = HEF(HEF_PATH)

    print("=== INPUT INFO ===")
    for info in hef.get_input_vstream_infos():
        print("name:", info.name, "| shape:", info.shape, "| format:", info.format)

    print("\n=== OUTPUT INFO ===")
    for info in hef.get_output_vstream_infos():
        print("name:", info.name, "| shape:", info.shape, "| format:", info.format)

    # [HW] hailort.service is active on this device, so every VDevice must use
    # the service (multi_process_service=True), including this standalone
    # diagnostic script. Otherwise, it may fail with
    # HAILO_OUT_OF_PHYSICAL_DEVICES because the service already manages the device.
    vdevice_params = VDevice.create_params()
    vdevice_params.group_id = "PROBE"  # Separate from group "1" used by the main pipeline.
    vdevice_params.multi_process_service = True
    vdevice_params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    with VDevice(vdevice_params) as target:
        configure_params = ConfigureParams.create_from_hef(
            hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_vstreams_params = InputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.UINT8)
        output_vstreams_params = OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=FormatType.FLOAT32)

        input_info = hef.get_input_vstream_infos()[0]
        h, w, c = input_info.shape

        # Random noise is used only to inspect the output structure; detection
        # accuracy and the semantic meaning of the predictions are not evaluated.
        dummy_img = np.random.randint(0, 255, size=(1, h, w, c), dtype=np.uint8)
        input_data = {input_info.name: dummy_img}

        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            with network_group.activate(network_group_params):
                results = infer_pipeline.infer(input_data)

        print("\n=== RAW RESULTS ===")
        print("type(results):", type(results))
        for key, val in results.items():
            print(f"\n--- output key: {key} ---")
            print("type(val):", type(val))
            if isinstance(val, np.ndarray):
                print("val.shape:", val.shape, "| val.dtype:", val.dtype)
                print("val[:2] (if available):", val[:2] if len(val) > 0 else val)
            elif isinstance(val, list):
                print("len(val):", len(val))
                for i, item in enumerate(val[:3]):
                    print(f"  val[{i}] type:", type(item))
                    if isinstance(item, np.ndarray):
                        print(f"  val[{i}] shape:", item.shape, "| dtype:", item.dtype)
                    elif isinstance(item, list):
                        print(f"  val[{i}] len:", len(item))
                        for j, sub in enumerate(item[:3]):
                            print(f"    val[{i}][{j}] type:", type(sub),
                                  "shape:", getattr(sub, "shape", None))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to convert PyTorch models to TorchScript format for C++ usage.
This converts the MiDaS and U2Net models used in the Fuser class.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import os


def convert_midas_model():
    """Convert MiDaS model to TorchScript"""
    print("Loading MiDaS model...")

    # Load MiDaS model
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
    midas.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, 384, 384)

    # Trace the model
    print("Tracing MiDaS model...")
    traced_model = torch.jit.trace(midas, dummy_input)

    # Save the traced model
    output_path = "midas_traced.pt"
    traced_model.save(output_path)
    print(f"MiDaS model saved to {output_path}")

    return output_path


def convert_u2net_model(model_path):
    """Convert U2Net model to TorchScript"""
    print("Loading U2Net model...")

    # You'll need to import your U2Net model class
    # This is a placeholder - adjust based on your actual U2Net implementation
    try:
        from data.models.u2net import U2NET

        # Initialize model
        model = U2NET(3, 1)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 3, 320, 320)

        # Trace the model
        print("Tracing U2Net model...")
        traced_model = torch.jit.trace(model, dummy_input)

        # Save the traced model
        output_path = "u2net_traced.pt"
        traced_model.save(output_path)
        print(f"U2Net model saved to {output_path}")

        return output_path

    except ImportError:
        print("Could not import U2NET class. Please ensure the model file is accessible.")
        return None


def test_traced_models(midas_path, u2net_path):
    """Test the traced models"""
    print("Testing traced models...")

    # Test MiDaS
    try:
        midas_model = torch.jit.load(midas_path)
        test_input = torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            output = midas_model(test_input)
        print(f"MiDaS test passed - Output shape: {output.shape}")
    except Exception as e:
        print(f"MiDaS test failed: {e}")

    # Test U2Net
    if u2net_path:
        try:
            u2net_model = torch.jit.load(u2net_path)
            test_input = torch.randn(1, 3, 320, 320)
            with torch.no_grad():
                output = u2net_model(test_input)
            print(f"U2Net test passed - Output shape: {output.shape}")
        except Exception as e:
            print(f"U2Net test failed: {e}")


def main():
    if len(sys.argv) > 1:
        u2net_model_path = sys.argv[1]
    else:
        # Default path from your original code
        u2net_model_path = "../data/models/u2net.pth"

    print("Converting PyTorch models to TorchScript...")
    print("This may take a few minutes for the first run (downloading models)...")

    # Convert models
    midas_traced = convert_midas_model()
    u2net_traced = convert_u2net_model(u2net_model_path) if os.path.exists(u2net_model_path) else None

    # Test the models
    test_traced_models(midas_traced, u2net_traced)

    print("\nConversion complete!")
    print("You can now use these traced models in your C++ application:")
    print(f"  MiDaS: {midas_traced}")
    if u2net_traced:
        print(f"  U2Net: {u2net_traced}")
    else:
        print("  U2Net: Conversion failed - check model path")


if __name__ == "__main__":
    main()
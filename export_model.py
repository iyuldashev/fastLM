import struct
import random

# CONFIGURATION
d_model = 4
layers = 1  # Just one Transformer Block
filename = "models/model.bin"

print(f"💾 Creating dummy model file: {filename}...")

# Open file in "Write Binary" (wb) mode
with open(filename, "wb") as f:
    # 1. Write Header (Magic Number to identify file)
    # We use 0xFEEDBEEF (A common engineering joke/standard)
    f.write(struct.pack("I", 0xFEEDBEEF)) 
    
    # 2. Write Dimensions
    f.write(struct.pack("ii", layers, d_model))

    # 3. Write Weights
    # We need 4 matrices (W_q, W_k, W_v, W_out)
    # Each matrix is (d_model x d_model) -> 4x4 = 16 floats
    # Total floats = 4 matrices * 16 floats = 64 floats
    
    print("  > Writing weights...")
    total_weights = 4 * (d_model * d_model)
    for _ in range(total_weights):
        # Generate random float
        val = random.random() 
        # Write as 'float' (4 bytes)
        f.write(struct.pack("f", val))

print("✅ Model saved successfully!")
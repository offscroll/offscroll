#!/bin/bash
# Create a minimal placeholder JPEG using Python
python3 << 'EOF'
import struct
import os

# Create a minimal valid JPEG file (800x600, light blue)
# This uses the minimal JPEG structure
def create_minimal_jpeg(width=800, height=600):
    """Create a minimal valid JPEG file."""
    jpeg_data = bytearray()

    # SOI (Start of Image)
    jpeg_data.extend(b'\xFF\xD8')

    # APP0 (JFIF header)
    jpeg_data.extend(b'\xFF\xE0')
    jpeg_data.extend(b'\x00\x10')  # Length
    jpeg_data.extend(b'JFIF\x00')  # Identifier
    jpeg_data.extend(b'\x01\x01')  # Version
    jpeg_data.extend(b'\x00')      # Units
    jpeg_data.extend(b'\x00\x01\x00\x01')  # X and Y density
    jpeg_data.extend(b'\x00\x00')  # Thumbnail dimensions

    # SOF0 (Start of Frame, Baseline DCT)
    jpeg_data.extend(b'\xFF\xC0')
    jpeg_data.extend(b'\x00\x11')  # Length (17 bytes)
    jpeg_data.extend(b'\x08')      # Precision
    jpeg_data.extend(struct.pack('>HH', height, width))  # Height, Width
    jpeg_data.extend(b'\x03')      # Components (3 for RGB)
    jpeg_data.extend(b'\x01\x11\x00')  # Component 1 (Y)
    jpeg_data.extend(b'\x02\x11\x01')  # Component 2 (Cb)
    jpeg_data.extend(b'\x03\x11\x01')  # Component 3 (Cr)

    # DHT (Define Huffman Table) - DC table for luminance
    jpeg_data.extend(b'\xFF\xC4')
    jpeg_data.extend(b'\x00\x1F')  # Length
    jpeg_data.extend(b'\x00')      # Table info
    jpeg_data.extend(b'\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00')
    jpeg_data.extend(b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B')

    # DHT - AC table for luminance
    jpeg_data.extend(b'\xFF\xC4')
    jpeg_data.extend(b'\x00\xB5')  # Length
    jpeg_data.extend(b'\x10')      # Table info
    jpeg_data.extend(b'\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01\x7D')
    jpeg_data.extend(b'\x01\x02\x03\x00\x04\x11\x05\x12\x21\x31\x41\x06\x13\x51\x61\x07')
    jpeg_data.extend(b'\x22\x71\x14\x32\x81\x91\xA1\x08\x23\x42\xB1\xC1\x15\x52\xD1\xF0')
    jpeg_data.extend(b'\x24\x33\x62\x72\x82\x09\x0A\x16\x17\x18\x19\x1A\x25\x26\x27\x28')
    jpeg_data.extend(b'\x29\x2A\x34\x35\x36\x37\x38\x39\x3A\x43\x44\x45\x46\x47\x48\x49')
    jpeg_data.extend(b'\x4A\x53\x54\x55\x56\x57\x58\x59\x5A\x63\x64\x65\x66\x67\x68\x69')
    jpeg_data.extend(b'\x6A\x73\x74\x75\x76\x77\x78\x79\x7A\x83\x84\x85\x86\x87\x88\x89')
    jpeg_data.extend(b'\x8A\x92\x93\x94\x95\x96\x97\x98\x99\x9A\xA2\xA3\xA4\xA5\xA6\xA7')
    jpeg_data.extend(b'\xA8\xA9\xAA\xB2\xB3\xB4\xB5\xB6\xB7\xB8\xB9\xBA\xC2\xC3\xC4\xC5')
    jpeg_data.extend(b'\xC6\xC7\xC8\xC9\xCA\xD2\xD3\xD4\xD5\xD6\xD7\xD8\xD9\xDA\xE1\xE2')
    jpeg_data.extend(b'\xE3\xE4\xE5\xE6\xE7\xE8\xE9\xEA\xF1\xF2\xF3\xF4\xF5\xF6\xF7\xF8')
    jpeg_data.extend(b'\xF9\xFA')

    # DQT (Define Quantization Table)
    jpeg_data.extend(b'\xFF\xDB')
    jpeg_data.extend(b'\x00\x43')  # Length
    jpeg_data.extend(b'\x00')      # Table info
    jpeg_data.extend(b'\x10\x0B\x0C\x0E\x0C\x0A\x10\x0E\x0D\x0E\x12\x11\x10\x13\x18\x28')
    jpeg_data.extend(b'\x1A\x18\x16\x16\x18\x31\x23\x25\x1D\x28\x3A\x33\x3D\x3C\x39\x33')
    jpeg_data.extend(b'\x38\x37\x40\x48\x5C\x4E\x40\x44\x57\x45\x37\x38\x50\x6D\x51\x57')
    jpeg_data.extend(b'\x5F\x62\x67\x68\x67\x3E\x4D\x71\x79\x70\x64\x78\x5C\x65\x67\x63')

    # SOS (Start of Scan)
    jpeg_data.extend(b'\xFF\xDA')
    jpeg_data.extend(b'\x00\x0C')  # Length
    jpeg_data.extend(b'\x03')      # Components
    jpeg_data.extend(b'\x01\x00')  # Component 1
    jpeg_data.extend(b'\x02\x11')  # Component 2
    jpeg_data.extend(b'\x03\x11')  # Component 3
    jpeg_data.extend(b'\x00\x3F\x00')  # Spectral selection

    # Minimal compressed image data (creates a uniform light blue image)
    jpeg_data.extend(b'\xFF\x00' * 20)  # Minimal MCU data

    # EOI (End of Image)
    jpeg_data.extend(b'\xFF\xD9')

    return bytes(jpeg_data)

# Create and save
jpeg_bytes = create_minimal_jpeg(800, 600)
with open('placeholder.jpg', 'wb') as f:
    f.write(jpeg_bytes)
print(f"Created placeholder.jpg ({len(jpeg_bytes)} bytes)")
EOF

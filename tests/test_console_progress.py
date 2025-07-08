#!/usr/bin/env python3
"""
Quick test to verify console progress logging works.
"""

import logging
import sys

sys.path.insert(0, "/Users/j/src/inferswitch")

from inferswitch.utils.logging import log_streaming_progress

# Configure logging to see console output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Test the progress logging
print("Testing console progress logging...")
print("You should see progress messages below:")
print("-" * 60)

# Simulate progress updates
log_streaming_progress(30.5, 1523, "claude-3-5-sonnet-20241022")
print("\n")
log_streaming_progress(65.2, 3102, "claude-3-5-sonnet-20241022")
print("\n")
log_streaming_progress(95.8, 4587, "claude-3-5-sonnet-20241022")
print("\n")

print("-" * 60)
print("Progress logging test complete!")
print("Check requests.log for file entries as well.")

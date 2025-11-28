"""
Clear all pending reviews from the Human Review queue
"""

import json
import os

review_queue_file = "review_queue.json"

if os.path.exists(review_queue_file):
    # Clear the queue
    with open(review_queue_file, 'w') as f:
        json.dump([], f)
    print(f"✅ Successfully cleared all pending reviews from {review_queue_file}")
else:
    print(f"ℹ️ No review queue file found at {review_queue_file}")
    print("Creating empty queue file...")
    with open(review_queue_file, 'w') as f:
        json.dump([], f)
    print("✅ Created empty review queue file")

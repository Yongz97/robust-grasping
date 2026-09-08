"""Decode the demo end to end and create a contact sheet for visual QA."""
import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("video", type=Path)
args = parser.parse_args()
reader = imageio.get_reader(str(args.video))
metadata = reader.get_meta_data()
times = [2.0, 5.7, 9.5, 12.0, 14.7, 19.4]
indices = [round(t * metadata["fps"]) for t in times]
frames = {}
count = 0
for index, frame in enumerate(reader):
    assert frame.shape == (640, 960, 3)
    if index in indices:
        frames[index] = frame
    count += 1
reader.close()
assert len(frames) == len(times), "Video ended before expected recovery sequence"
sheet = np.zeros((3 * 352, 2 * 480, 3), dtype=np.uint8)
for i, (index, frame) in enumerate(frames.items()):
    x, y = (i % 2) * 480, (i // 2) * 352
    sheet[y:y + 320, x:x + 480] = cv2.resize(frame, (480, 320))
    cv2.putText(sheet, f"{index / metadata['fps']:.1f}s", (x + 12, y + 342),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
output = args.video.with_name("video-review.png")
cv2.imwrite(str(output), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
print(f"Decoded {count} frames at {metadata['fps']} fps ({count / metadata['fps']:.2f}s)")
print(output)

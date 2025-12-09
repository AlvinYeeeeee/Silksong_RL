"""
Test the anti-stuck menu detection.
Run this while on the boss menu screen to see what's detected.
"""

import time
import numpy as np
from mss.windows import MSS as mss
import cv2

# Get monitor region (same as LaceEnv)
def get_monitor():
    import pyautogui
    windows = pyautogui.getWindowsWithTitle('Hollow Knight')
    for w in windows:
        if 'silksong' in w.title.lower():
            TITLE_BAR_HEIGHT = 32
            return {
                'left': w.left,
                'top': w.top + TITLE_BAR_HEIGHT,
                'width': 1280,
                'height': 720
            }
    raise RuntimeError("Silksong window not found!")

print("=" * 60)
print("ANTI-STUCK DETECTION TEST")
print("=" * 60)
print("\nMake sure you're on the BOSS MENU screen (white Hornet)!")
print("This will test detection every 0.5 seconds for 30 seconds.")
print()

input("Press ENTER to start...")

monitor = get_monitor()
print(f"Monitor: {monitor}")

print("\nStarting detection loop...")
print("Box region: (610, 320) to (679, 345)")
print()

start_time = time.time()
white_start = None

while time.time() - start_time < 30:
    with mss() as sct:
        frame = np.asarray(sct.grab(monitor), dtype=np.uint8)
    
    # Check specific box region
    check_box = frame[320:345, 610:679, :3]  # y1:y2, x1:x2
    
    # Stats about the box
    min_val = check_box.min()
    max_val = check_box.max()
    mean_val = check_box.mean()
    
    # Check if all white
    all_white = (check_box > 240).all()
    
    # Track time white
    if all_white:
        if white_start is None:
            white_start = time.time()
        white_duration = time.time() - white_start
    else:
        white_start = None
        white_duration = 0
    
    status = "ALL WHITE!" if all_white else "not white"
    print(f"Box: min={min_val:3d}, max={max_val:3d}, mean={mean_val:6.1f} | {status} | white for {white_duration:.1f}s", end='\r')
    
    # Save screenshot if requested
    if all_white and white_duration > 2.0:
        print(f"\n>>> WOULD PRESS B NOW! (white for {white_duration:.1f}s)")
        white_start = time.time()  # Reset
    
    time.sleep(0.5)

print("\n\nTest complete!")

# Save a debug screenshot
print("\nSaving debug screenshot with box highlighted...")
with mss() as sct:
    frame = np.asarray(sct.grab(monitor), dtype=np.uint8)

# Draw rectangle on the check area
cv2.rectangle(frame, (610, 320), (679, 345), (0, 255, 0), 2)
cv2.imwrite('antistuck_debug.png', frame)
print("Saved: antistuck_debug.png")
print("The green box shows the detection region.")


"""
Test script to verify the game controls are working.
Run this with the game open and watch if the character moves!

Uses pydirectinput which works better with games than pyautogui.
"""

import time

try:
    import pydirectinput
    pydirectinput.FAILSAFE = False
    pydirectinput.PAUSE = 0.
    print("Using pydirectinput (DirectInput - works with games)")
except ImportError:
    print("pydirectinput not installed! Run: pip install pydirectinput")
    exit(1)

def test_movement():
    """Test basic movement keys."""
    print("\n" + "=" * 50)
    print("SILKSONG CONTROL TEST")
    print("=" * 50)
    print("\nMake sure:")
    print("  1. Silksong is running")
    print("  2. You are IN A FIGHT or can move freely")
    print("  3. Game window is visible")
    print()
    
    input("Press ENTER when ready, then quickly click on the game window!")
    
    print("\nStarting in 3 seconds - CLICK ON GAME WINDOW NOW!")
    time.sleep(3)
    
    print("\n--- Testing Movement ---")
    
    print("Pressing 'a' (move left) for 0.5 seconds...")
    pydirectinput.keyDown('a')
    time.sleep(0.5)
    pydirectinput.keyUp('a')
    time.sleep(0.3)
    
    print("Pressing 'd' (move right) for 0.5 seconds...")
    pydirectinput.keyDown('d')
    time.sleep(0.5)
    pydirectinput.keyUp('d')
    time.sleep(0.3)
    
    print("Pressing 'space' (jump)...")
    pydirectinput.press('space')
    time.sleep(0.5)
    
    print("Pressing 'k' (attack)...")
    pydirectinput.press('k')
    time.sleep(0.3)
    
    print("Pressing 'j' (spell)...")
    pydirectinput.press('j')
    time.sleep(0.3)
    
    print("Pressing 'l' (dash)...")
    pydirectinput.press('l')
    time.sleep(0.3)
    
    print("\n" + "=" * 50)
    print("TEST COMPLETE")
    print("=" * 50)
    print("\nDid the character move/attack/jump?")
    print("  YES -> Great! Training should work!")
    print("  NO  -> Check your key bindings in Silksong settings")

if __name__ == '__main__':
    test_movement()

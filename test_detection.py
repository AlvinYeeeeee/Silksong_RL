"""
Manual Play Detection Test

This script lets YOU play the game while monitoring all the detection logic:
- Hornet HP
- Silk amount
- Boss HP
- Win/Lose conditions
- Blackscreen detection
- Anti-stuck detection

Use this to verify that victory/defeat detection is working correctly!

Usage:
    python test_detection.py

Controls:
    - You play the game normally
    - Watch the console for live detection output
    - Press Ctrl+C to stop
"""

import time
import numpy as np
from mss import mss
import pydirectinput

import lace_env

def main():
    print("=" * 60)
    print("🎮 MANUAL PLAY DETECTION TEST")
    print("=" * 60)
    print()
    print("This will monitor the game while YOU play.")
    print("Watch the console to see if detection is working!")
    print()
    print("Switch to the game window now...")
    for i in range(5, 0, -1):
        print(f"  Starting in {i}...")
        time.sleep(1)
    print("  GO! Start playing!\n")
    print("=" * 60)
    
    # Create environment (but we won't use step(), just observe())
    env = lace_env.LaceEnvV2(
        obs_shape=(192, 192),
        rgb=False,
        gap=0.17,
    )
    
    # Initialize tracking variables (same as env.prepare())
    prev_hornet_hp = 9
    prev_enemy_hp = 1.0
    prev_silk = 18
    boss_hp_bar_seen = False
    boss_hp_low_seen = False
    hp_bar_disappeared_time = None
    hornet_zero_hp_time = None
    episode_start = time.time()
    
    last_print_time = 0
    print_interval = 0.2  # Print every 0.2 seconds
    
    # Track the LOWEST boss HP we've seen (boss HP can never go UP)
    lowest_enemy_hp = 1.0
    
    try:
        while True:
            # Get observation
            obs, hornet_hp, raw_enemy_hp, silk_amount, hp_bar_visible, is_blackscreen = env.observe()
            
            current_time = time.time()
            
            # SANITY CHECK: Boss HP can NEVER increase!
            # If we see it go up, that's a false detection (white flash, etc.)
            impossible_reading = raw_enemy_hp > lowest_enemy_hp + 0.01
            
            if impossible_reading:
                # This is impossible - boss HP went UP
                # Keep using the lowest HP we've seen
                enemy_hp = lowest_enemy_hp
                # Don't spam the console, just note it occasionally
            else:
                enemy_hp = raw_enemy_hp
                if enemy_hp < lowest_enemy_hp:
                    lowest_enemy_hp = enemy_hp
            
            # Track boss HP bar visibility
            # BUT: If we got an impossible reading, that's NOT a real HP bar - ignore it!
            real_hp_bar_visible = hp_bar_visible and not impossible_reading
            
            if real_hp_bar_visible and not boss_hp_bar_seen:
                boss_hp_bar_seen = True
                print("\n>>> Boss HP bar appeared! Win/lose detection now active.")
            
            # Track if boss HP ever dropped below 5%
            if lowest_enemy_hp < 0.05 and not boss_hp_low_seen:
                boss_hp_low_seen = True
                print("\n>>> Boss HP dropped below 5%! Victory condition can now trigger.")
            
            # Track HP bar disappearance (for victory)
            # KEY FIX: Impossible readings (white flashes) should NOT reset the disappearance timer!
            if boss_hp_bar_seen:
                if (not hp_bar_visible or impossible_reading) and not is_blackscreen:
                    # HP bar is gone OR we're seeing false readings (white flash)
                    if hp_bar_disappeared_time is None:
                        hp_bar_disappeared_time = current_time
                        print("\n>>> HP bar disappeared! Tracking for victory...")
                elif hp_bar_visible and not impossible_reading:
                    # HP bar is REALLY visible with valid reading
                    if hp_bar_disappeared_time is not None:
                        print("\n>>> HP bar reappeared (real).")
                    hp_bar_disappeared_time = None
            
            # Track Hornet HP = 0 (for defeat)
            if hornet_hp == 0:
                if hornet_zero_hp_time is None:
                    hornet_zero_hp_time = current_time
                    print("\n>>> Hornet HP hit 0! Tracking for defeat...")
            else:
                if hornet_zero_hp_time is not None:
                    print("\n>>> Hornet HP recovered (heal or false detection)")
                hornet_zero_hp_time = None
            
            # Check win condition
            win = False
            lose = False
            
            if boss_hp_bar_seen and (current_time - episode_start) > 5.0:
                # Defeat: Hornet HP = 0 for > 1.5s
                if hornet_zero_hp_time is not None:
                    time_at_zero = current_time - hornet_zero_hp_time
                    if time_at_zero > 1.5:
                        lose = True
                        print(f"\n{'='*60}")
                        print(f"💀 DEFEAT DETECTED!")
                        print(f"   Hornet HP was 0 for {time_at_zero:.1f}s")
                        print(f"{'='*60}\n")
                
                # Victory: Boss HP was <5%, HP bar gone >3s, Hornet alive
                if boss_hp_low_seen and hp_bar_disappeared_time is not None and hornet_hp > 0:
                    time_gone = current_time - hp_bar_disappeared_time
                    if time_gone > 3.0:
                        win = True
                        print(f"\n{'='*60}")
                        print(f"🏆 VICTORY DETECTED!")
                        print(f"   Boss HP was <5%, HP bar gone {time_gone:.1f}s, Hornet HP={hornet_hp}")
                        print(f"{'='*60}\n")
            
            # Periodic status print
            if current_time - last_print_time > print_interval:
                last_print_time = current_time
                
                # Calculate changes
                hp_change = hornet_hp - prev_hornet_hp
                enemy_change = enemy_hp - prev_enemy_hp
                silk_change = silk_amount - prev_silk
                
                hp_str = f"HP: {hornet_hp}/9"
                if hp_change != 0:
                    hp_str += f" ({'+' if hp_change > 0 else ''}{hp_change})"
                
                silk_str = f"Silk: {silk_amount}/18"
                if silk_change != 0:
                    silk_str += f" ({'+' if silk_change > 0 else ''}{silk_change})"
                
                boss_str = f"Boss: {enemy_hp*100:.1f}% (low:{lowest_enemy_hp*100:.1f}%)"
                if enemy_change != 0:
                    boss_str += f" Δ{'+' if enemy_change > 0 else ''}{enemy_change*100:.1f}%"
                
                # Status indicators
                status = []
                if is_blackscreen:
                    status.append("⬛BLACKSCREEN")
                if not hp_bar_visible:
                    status.append("👁️NO_HP_BAR")
                if boss_hp_bar_seen:
                    status.append("✅BOSS_SEEN")
                if boss_hp_low_seen:
                    status.append("⚡BOSS_LOW")
                
                # Time tracking
                time_info = []
                if hp_bar_disappeared_time:
                    time_info.append(f"bar_gone:{current_time - hp_bar_disappeared_time:.1f}s")
                if hornet_zero_hp_time:
                    time_info.append(f"hp0:{current_time - hornet_zero_hp_time:.1f}s")
                
                # Build output line
                line = f"\r{hp_str:<20} {silk_str:<20} {boss_str:<20}"
                if status:
                    line += f" [{' '.join(status)}]"
                if time_info:
                    line += f" ({', '.join(time_info)})"
                
                print(line, end='', flush=True)
                
                prev_hornet_hp = hornet_hp
                prev_enemy_hp = enemy_hp
                prev_silk = silk_amount
            
            # If win/lose detected, auto-reset like the real environment
            if win or lose:
                wait_time = 10 if win else 4  # Victory animation is longer
                print(f"\n>>> Auto-resetting in {wait_time} seconds...")
                time.sleep(wait_time)
                
                # Press b + space + space (menu navigation)
                print(">>> Pressing: b")
                pydirectinput.keyDown('b')
                time.sleep(0.1)
                pydirectinput.keyUp('b')
                
                print(">>> Pressing: space")
                pydirectinput.keyDown('space')
                time.sleep(0.1)
                pydirectinput.keyUp('space')
                
                print(">>> Pressing: space")
                pydirectinput.keyDown('space')
                time.sleep(0.1)
                pydirectinput.keyUp('space')
                
                # Wait for loading
                print(">>> Waiting 2 seconds for load...")
                time.sleep(2)
                
                # Dash into arena (d + l)
                print(">>> Dashing into arena: d + l")
                pydirectinput.keyDown('d')
                time.sleep(0.05)
                pydirectinput.keyDown('l')
                time.sleep(0.1)
                pydirectinput.keyUp('l')
                pydirectinput.keyUp('d')
                
                # Wait for fight to start
                print(">>> Waiting 1 second...")
                time.sleep(1)
                
                # Reset tracking
                prev_hornet_hp = 9
                prev_enemy_hp = 1.0
                prev_silk = 18
                boss_hp_bar_seen = False
                boss_hp_low_seen = False
                hp_bar_disappeared_time = None
                hornet_zero_hp_time = None
                lowest_enemy_hp = 1.0  # Reset lowest HP tracker
                episode_start = time.time()
                print("\n>>> Tracking reset! Next fight starting...\n")
            
            # Small delay to not spam CPU
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    
    env.close()
    print("Done!")


if __name__ == '__main__':
    main()


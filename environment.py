"""
Game Environment for Hollow Knight: Silksong - Lace Boss Fight

This module provides a Gym-compatible environment that interfaces with
Silksong to train an RL agent. It handles:
- Screen capture and observation preprocessing
- Keyboard input for game control
- HP/Silk resource detection via pixel analysis
- Victory/defeat condition detection
- Reward calculation for training
"""

import gc
import gym
import cv2
import time
import enum
import random
import pyautogui
import pydirectinput  # For game input (DirectInput)
import threading
import numpy as np
from mss.windows import MSS as mss

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.
pydirectinput.FAILSAFE = False
pydirectinput.PAUSE = 0.


# =============================================================================
# ACTION DEFINITIONS
# =============================================================================

class Actions(enum.Enum):
    @classmethod
    def random(cls):
        return random.choice(list(cls))


class Move(Actions):
    """Movement actions"""
    NO_OP = 0
    HOLD_LEFT = 1
    HOLD_RIGHT = 2


class Attack(Actions):
    """Attack actions - includes basic attack, spell, and tool"""
    NO_OP = 0
    ATTACK = 1      # Basic attack (k)
    SPELL = 2       # Spell attack (j)
    TOOL = 3        # Tool attack (w+j combo)


class Displacement(Actions):
    """Jump and dash actions"""
    NO_OP = 0
    TIMED_SHORT_JUMP = 1
    TIMED_LONG_JUMP = 2
    DASH = 3


class Heal(Actions):
    """Heal action - uses silk resource"""
    NO_OP = 0
    HEAL = 1        # Heal (Lshift) - costs silk


# =============================================================================
# ENVIRONMENT CLASS
# =============================================================================

class LaceEnv(gym.Env):
    """
    Environment that interacts with Hollow Knight: Silksong game,
    specifically for the Lace boss fight.
    
    Implementation follows the gym custom environment API.
    
    Lace is a single-phase boss, so no phase tracking is needed.
    """

    # =========================================================================
    # KEYMAPS - Silksong controls
    # =========================================================================
    KEYMAPS = {
        Move.HOLD_LEFT: 'a',
        Move.HOLD_RIGHT: 'd',
        Displacement.TIMED_SHORT_JUMP: 'space',
        Displacement.TIMED_LONG_JUMP: 'space',
        Displacement.DASH: 'l',
        Attack.ATTACK: 'k',         # Basic attack
        Attack.SPELL: 'j',          # Spell
        Attack.TOOL: ('w', 'j'),    # Tool = w+j combo
        Heal.HEAL: 'shiftleft',     # Heal - costs silk
    }
    
    REWMAPS = {
        Move.HOLD_LEFT: 0,
        Move.HOLD_RIGHT: 0,
        Displacement.TIMED_SHORT_JUMP: 0,
        Displacement.TIMED_LONG_JUMP: 0,
        Displacement.DASH: 0,
        Attack.ATTACK: 0,
        Attack.SPELL: 0,
        Attack.TOOL: 0,
        Heal.HEAL: 0,
    }

    # =========================================================================
    # TODO: Update these HP checkpoint positions for Silksong's health beads
    # These are X-pixel positions where each health bead is located
    # You need to take a screenshot and measure where each bead is
    # =========================================================================
    HP_CKPT = np.array([
        144, 177, 210, 243, 276, 309, 342, 375, 408, 441  
    ], dtype=int)
    
    HP_BAR_Y = 68
    
    # =========================================================================
    # SILK METER DETECTION
    # The silk meter is a horizontal bar with 18 units (similar to HP beads)
    # TODO: Calibrate these X positions for each silk unit
    # =========================================================================
    SILK_BAR_Y = 128          # Y coordinate of the silk bar row
    SILK_CKPT = np.array([    # X coordinates for each of the 18 silk units
        79, 88, 93, 100, 107, 113, 119, 125, 132, 139, 145, 151, 158, 164, 170, 177, 183, 189
    ], dtype=int)
    SILK_THRESHOLD = 150      # Pixel brightness threshold (silk is bright, empty is dark)
    
    # Silk costs for actions (adjust based on Silksong mechanics)
    SILK_COST_SPELL = 4       # Silk needed to cast spell
    SILK_COST_HEAL = 9        # Silk needed to heal
    
    # Tool constraints
    TOOL_COOLDOWN = 5.0       # Cooldown between tool uses (seconds)

    ACTIONS = [Move, Attack, Displacement, Heal]

    def __init__(self, obs_shape=(160, 160), rgb=False, gap=0.165,
                 damage_penalty=0.5, hit_reward=0.85, heal_reward=0.7,
                 inactivity_penalty=0.3, inactivity_window=5.0,
                 victory_base=10.0, defeat_base=8.0):
        """
        Initialize the Lace environment.
        
        :param obs_shape: the shape of observation returned by step and reset
        :param rgb: if True, return RGB observations; else grayscale
        :param gap: minimum time between actions (controls action rate ~6 FPS)
        :param damage_penalty: penalty when Hornet takes damage (positive value, will be negated)
        :param hit_reward: reward for hitting boss
        :param heal_reward: reward for successful heal
        :param inactivity_penalty: penalty for not dealing damage (applied every inactivity_window seconds)
        :param inactivity_window: seconds without dealing damage before penalty kicks in
        :param victory_base: base reward for winning (NOT clipped)
        :param defeat_base: base penalty for losing (NOT clipped, positive value will be negated)
        """
        self.monitor = self._find_window()
        self.holding = []
        self.prev_hornet_hp = None
        self.prev_enemy_hp = None
        self.prev_silk = None
        self.prev_action = -1
        self._first_reset = True  # Track if this is the first reset (user already in fight)
        self._hp_bar_disappeared_time = None  # Track when HP bar disappeared
        self._boss_hp_bar_seen = False  # Track if HP bar was ever visible this episode
        self._boss_hp_low_seen = False  # Track if boss HP ever dropped below 5% (for victory condition)
        self._tool_uses = 0  # Track tool usage (max 8 per episode)
        self.MAX_TOOL_USES = 8
        self._last_tool_time = None  # Track cooldown for tool usage
        self._hornet_zero_hp_time = None  # Track when hornet HP first became 0 (anti-stuck)
        self._menu_stuck_time = None  # Track when menu screen detected (anti-stuck)
        self._last_damage_time = None  # Track when we last dealt damage (inactivity detection)
        self._last_move_time = None  # Track when we last moved left/right
        self._last_displacement_time = None  # Track when we last jumped/dashed
        self._lowest_enemy_hp = 1.0  # Track lowest boss HP seen (boss HP can never go UP)
        self._last_episode_was_victory = False  # Track if last episode ended in victory (for reset timing)
        self._blackscreen_start_time = None  # Track when blackscreen started (for defeat detection)
        self._blackscreen_punished = False  # Track if we already punished this blackscreen session
        self._timeout_occurred = False  # Track if episode ended due to timeout
        self._last_hurt_time = None  # Track when we last took damage (for combo damage penalty)
        self._episode_time = time.time()  # Track episode start time (for timeout detection)
        
        total_actions = np.prod([len(Act) for Act in self.ACTIONS])
        if rgb:
            obs_shape = (3,) + obs_shape
        else:
            obs_shape = (1,) + obs_shape
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                dtype=np.uint8, shape=obs_shape)
        self.action_space = gym.spaces.Discrete(int(total_actions))
        self.rgb = rgb
        self.gap = gap
        self._prev_time = None

        # Reward parameters
        self.damage_penalty = damage_penalty
        self.hit_reward = hit_reward
        self.heal_reward = heal_reward
        self.inactivity_penalty = inactivity_penalty
        self.inactivity_window = inactivity_window
        self.victory_base = victory_base
        self.defeat_base = defeat_base

        self._hold_time = 0.2
        self._fail_hold_rew = -1e-4
        self._timer = None
        self._episode_time = None

    # =========================================================================
    # WINDOW DETECTION
    # =========================================================================
    
    @staticmethod
    def _find_window():
        """
        Find the location of Silksong window and return capture region.
        
        Simple approach: Find window by title, move to (0,0), use fixed offsets.
        No locator image needed!
        
        :return: monitor location dictionary for screenshot capture
        """
        window_title = 'Hollow Knight Silksong'
        
        window = pyautogui.getWindowsWithTitle(window_title)
        if len(window) == 0:
            # Fallback: try partial match
            all_windows = pyautogui.getAllWindows()
            silksong_windows = [w for w in all_windows if 'silksong' in w.title.lower()]
            if silksong_windows:
                window = silksong_windows
            else:
                raise RuntimeError(
                    f"Could not find Silksong window. "
                    f"Make sure the game is running. "
                    f"Searched for: '{window_title}'"
                )
        
        if len(window) == 0:
            raise RuntimeError("No Silksong window found!")
        
        # If multiple windows found, pick the one with actual size (the game window, not launcher)
        if len(window) > 1:
            print(f"Found {len(window)} windows, selecting the main game window...")
            # Filter to windows with reasonable size (game window should be large)
            sized_windows = [w for w in window if w.width > 500 and w.height > 300]
            if sized_windows:
                window = sized_windows[0]
            else:
                window = window[0]  # Fallback to first
        else:
            window = window[0]
        
        try:
            window.activate()
        except Exception:
            window.minimize()
            window.maximize()
            window.restore()
        window.moveTo(0, 0)
        
        # Simple approach: use window bounds directly
        # Assumes game is running at 1280x720 windowed mode
        # Title bar is typically ~32 pixels on Windows 10/11
        TITLE_BAR_HEIGHT = 32  # Adjust if needed for your system
        
        loc = {
            'left': window.left,
            'top': window.top + TITLE_BAR_HEIGHT,
            'width': 1280,   # Game width (1280x720 resolution)
            'height': 720    # Game height
        }
        
        print(f"Capture region: {loc}")
        return loc

    # =========================================================================
    # ACTION HANDLING
    # =========================================================================
    
    def _timed_hold(self, key, seconds):
        """
        Use a separate thread to hold a key for given seconds.
        Used for variable jump heights.
        
        :param key: the key to be pressed
        :param seconds: time to hold the key
        :return: 1 if already holding, 0 when success
        """
        def timer_thread():
            pydirectinput.keyDown(key)
            time.sleep(seconds)
            pydirectinput.keyUp(key)
            time.sleep(0.0005)

        if self._timer is None or not self._timer.is_alive():
            self._timer = threading.Thread(target=timer_thread)
            self._timer.start()
            return 0
        else:
            return 1

    def _focus_game_window(self):
        """Ensure the game window is focused before sending keys."""
        try:
            windows = pyautogui.getWindowsWithTitle('Hollow Knight')
            for w in windows:
                if 'silksong' in w.title.lower():
                    w.activate()
                    return
        except Exception:
            pass  # Window might already be focused

    def _step_actions(self, actions):
        """
        Release all non-timed holding keys, press keys for given actions.
        
        :param actions: a list of actions
        :return: reward for doing given actions
        """
        t = self.gap - (time.time() - self._prev_time)
        if t > 0:
            time.sleep(t)
        self._prev_time = time.time()
        
        # Make sure game window is focused
        self._focus_game_window()

        for key in self.holding:
            pydirectinput.keyUp(key)
        self.holding = []
        action_rew = 0
        
        for act in actions:
            if not act.value:
                continue
            key = self.KEYMAPS[act]
            action_rew += self.REWMAPS[act]

            if act.name.startswith('HOLD'):
                pydirectinput.keyDown(key)
                self.holding.append(key)
            elif act.name.startswith('TIMED'):
                action_rew += (self._fail_hold_rew *
                               self._timed_hold(key, act.value * self._hold_time))
            elif isinstance(key, tuple):
                # Combo key (like w+j for TOOL)
                pydirectinput.keyDown(key[0])
                time.sleep(0.05)
                pydirectinput.keyDown(key[1])
                time.sleep(0.1)
                pydirectinput.keyUp(key[1])
                time.sleep(0.05)
                pydirectinput.keyUp(key[0])
            else:
                # Use keyDown/keyUp instead of press() - press() doesn't work!
                pydirectinput.keyDown(key)
                time.sleep(0.1)
                pydirectinput.keyUp(key)
        return action_rew

    def _to_multi_discrete(self, num):
        """
        Interpret the single action number to a list of action enums.
        
        :param num: the number representing an action combination
        :return: list of action enums
        """
        num = int(num)
        chosen = []
        for Act in self.ACTIONS:
            num, mod = divmod(num, len(Act))
            chosen.append(Act(mod))
        return chosen

    # =========================================================================
    # MENU DETECTION
    # =========================================================================
    
    def _find_menu(self):
        """
        Locate the menu/challenge button for Lace.
        
        TODO: Create a screenshot of the UI element that indicates
        the player can start the fight.
        
        :return: the location of menu badge, or None if not found
        """
        monitor = self.monitor
        monitor = (monitor['left'] + monitor['width'] // 2,
                   monitor['top'] + monitor['height'] // 4,
                   monitor['width'] // 2,
                   monitor['height'] // 2)
        
        locator_path = './locator/challenge.png'
        
        try:
            return pyautogui.locateOnScreen(locator_path,
                                            region=monitor,
                                            confidence=0.925)
        except Exception:
            return None

    # =========================================================================
    # OBSERVATION AND HP EXTRACTION
    # =========================================================================
    
    def observe(self, force_gray=False):
        """
        Take a screenshot and identify enemy and player HP.
        
        TODO: Update HP extraction logic for Silksong:
        1. Find the Y-coordinate of Silksong's health bar
        2. Find the X-coordinates of each health bead
        3. Determine pixel color thresholds for full vs empty beads
        4. Update enemy HP bar detection for Lace
        
        :param force_gray: override self.rgb to force return gray obs
        :return: observation (resized screenshot), player HP, enemy HP
        """
        with mss() as sct:
            frame = np.asarray(sct.grab(self.monitor), dtype=np.uint8)
        
        # =====================================================================
        # SCREEN BRIGHTNESS (for blackscreen and win/lose detection)
        # =====================================================================
        screen_brightness = frame[:, :, 0].mean()
        is_blackscreen = screen_brightness < 30
        
        # =====================================================================
        # ENEMY HP EXTRACTION
        # HP bar: White background, gray health bar content
        # Dark/gray pixels = health, bright/white = empty (background)
        # HP bar won't appear until boss takes first damage!
        # =====================================================================
        
        enemy_hp_bar_y = 680
        enemy_hp_bar_x_start = 348
        enemy_hp_bar_x_end = 947
        
        # NOTE: x_end is INCLUSIVE (inside the bar), so we need +1 for Python slicing
        enemy_hp_bar = frame[enemy_hp_bar_y, enemy_hp_bar_x_start:enemy_hp_bar_x_end + 1, :]
        
        # Bar colors (BGRA format from mss):
        # Health color: #beb8b8ff → B=190, G=184, R=184 (grayish)
        # Background (empty): #ffffffff → B=255, G=255, R=255 (white)
        #
        # Health pixels: B channel around 190 (let's say 170-220)
        # Empty pixels: B channel around 255 (let's say > 240)
        
        b_channel = enemy_hp_bar[..., 0]
        total_pixels = len(enemy_hp_bar)
        
        # Count health pixels (grayish, B around 190)
        health_pixels = ((b_channel >= 170) & (b_channel <= 220)).sum()
        
        # Count empty pixels (white, B > 240)
        empty_pixels = (b_channel > 240).sum()
        
        # Bar is visible if most pixels are either health OR empty color
        # (not random colors from other UI elements)
        recognized_pixels = health_pixels + empty_pixels
        hp_bar_visible = recognized_pixels >= (total_pixels * 0.8)  # 80% must be health or empty
        
        if not hp_bar_visible:
            # HP bar not visible (either before first hit, or boss defeated)
            enemy_hp = self.prev_enemy_hp if self.prev_enemy_hp is not None else 1.0
        else:
            # HP bar visible - calculate health ratio
            # HP = health pixels / (health + empty pixels)
            if recognized_pixels > 0:
                enemy_hp = health_pixels / recognized_pixels
            else:
                enemy_hp = self.prev_enemy_hp if self.prev_enemy_hp is not None else 1.0
        
        # =====================================================================
        # PLAYER HP EXTRACTION
        # =====================================================================
        
        if is_blackscreen and self.prev_hornet_hp is not None:
            hornet_hp = self.prev_hornet_hp
        else:
            hornet_hp_bar = frame[self.HP_BAR_Y, :, 0]
            checkpoint1 = hornet_hp_bar[self.HP_CKPT]
            checkpoint2 = hornet_hp_bar[self.HP_CKPT - 1]
            hornet_hp = ((checkpoint1 > 200) | (checkpoint2 > 200)).sum()
        
        # =====================================================================
        # SILK METER EXTRACTION
        # Silk bar is horizontal with 18 units (similar to HP beads)
        # Bright pixels = silk filled, Dark pixels = empty
        # =====================================================================
        
        if is_blackscreen and self.prev_silk is not None:
            # During blackscreen, keep previous silk value
            silk_amount = self.prev_silk
        else:
            silk_bar = frame[self.SILK_BAR_Y, :, 0]  # Get row at SILK_BAR_Y, blue channel
            silk_values = silk_bar[self.SILK_CKPT]   # Sample at each silk unit position
            silk_amount = (silk_values > self.SILK_THRESHOLD).sum()  # Count filled units (0-18)
        
        # =====================================================================
        # OBSERVATION PROCESSING
        # =====================================================================
        
        rgb = not force_gray and self.rgb
        
        # Crop to arena (exclude bottom UI if needed)
        # For 720p: use most of the frame, maybe exclude bottom HP bar area
        arena_height = 700  # Adjust based on where boss HP bar is
        obs = cv2.cvtColor(frame[:arena_height, ...],
                           (cv2.COLOR_BGRA2RGB if rgb else cv2.COLOR_BGRA2GRAY))
        obs = cv2.resize(obs,
                         dsize=self.observation_space.shape[1:],
                         interpolation=cv2.INTER_AREA)
        
        # Make channel first (C, H, W)
        obs = np.rollaxis(obs, -1) if rgb else obs[np.newaxis, ...]
        
        return obs, hornet_hp, enemy_hp, silk_amount, hp_bar_visible, is_blackscreen

    # =========================================================================
    # STEP FUNCTION
    # =========================================================================
    
    def step(self, actions):
        """
        Execute one step in the environment.
        
        :param actions: action number to execute
        :return: (observation, reward, done, truncated, info)
        """
        action_rew = 0
        
        # Small penalty for repeating same action
        if actions == self.prev_action:
            action_rew -= 2e-5
        self.prev_action = actions
        
        # Convert action number to multi-discrete actions and execute
        actions = self._to_multi_discrete(actions)
        
        # Check which actions are being used
        move_action = actions[0]      # Move is index 0 in ACTIONS
        attack_action = actions[1]    # Attack is index 1 in ACTIONS
        disp_action = actions[2]      # Displacement is index 2 in ACTIONS
        heal_action = actions[3]      # Heal is index 3 in ACTIONS
        
        used_move = (move_action == Move.HOLD_LEFT or move_action == Move.HOLD_RIGHT)
        used_displacement = (disp_action != Displacement.NO_OP)  # Any jump or dash
        used_spell = (attack_action == Attack.SPELL)
        used_tool = (attack_action == Attack.TOOL)
        used_heal = (heal_action == Heal.HEAL)
        
        # Track movement and displacement for inactivity penalties
        if used_move:
            self._last_move_time = time.time()
        if used_displacement:
            self._last_displacement_time = time.time()
        
        # Track and limit tool usage (max 8 per episode, with 5s cooldown)
        if used_tool:
            current_time = time.time()
            # Check if tool is on cooldown
            if self._last_tool_time is not None:
                time_since_tool = current_time - self._last_tool_time
                if time_since_tool < self.TOOL_COOLDOWN:
                    action_rew -= 0.3  # Penalty for trying to use tool during cooldown
                    actions[1] = Attack.NO_OP  # Replace with no-op
                    used_tool = False  # Mark as not used
            
            # Check if tool uses depleted (only if not already blocked by cooldown)
            if used_tool and self._tool_uses >= self.MAX_TOOL_USES:
                action_rew -= 0.5  # Heavy penalty for trying to use tool when depleted
                actions[1] = Attack.NO_OP  # Don't execute the tool action
                used_tool = False
            elif used_tool:
                # Tool use allowed - increment counter and set cooldown
                self._tool_uses += 1
                self._last_tool_time = current_time
        
        # Penalize using spell/heal without enough silk
        if used_spell and self.prev_silk < self.SILK_COST_SPELL:
            action_rew -= 0.3  # Wasted spell attempt
        if used_heal and self.prev_silk < self.SILK_COST_HEAL:
            action_rew -= 0.3  # Wasted heal attempt
        
        # Penalize using spell when HP is low (save silk for healing!)
        if used_spell and self.prev_hornet_hp <= 3:
            action_rew -= 0.2  # Don't waste silk when you need to heal!
        
        # Debug: print live status
        action_names = [a.name for a in actions]
        move, attack, disp, heal = action_names
        print(f"HP:{self.prev_hornet_hp or 0:2d} | Silk:{self.prev_silk or 0:2d} | Boss:{(self.prev_enemy_hp or 1)*100:5.1f}% | {move:11s} {attack:6s} {disp:16s} {heal:4s}", end='\r')
        
        action_rew += self._step_actions(actions)
        
        # Get observation and HP values
        obs, hornet_hp, raw_enemy_hp, silk_amount, hp_bar_visible, is_blackscreen = self.observe()
        
        # SANITY CHECK: Boss HP can NEVER increase!
        # If we see it go up, that's a false detection (white flash during victory animation)
        impossible_reading = raw_enemy_hp > self._lowest_enemy_hp + 0.01
        
        if impossible_reading:
            # Keep using the lowest HP we've seen
            enemy_hp = self._lowest_enemy_hp
        else:
            enemy_hp = raw_enemy_hp
            if enemy_hp < self._lowest_enemy_hp:
                self._lowest_enemy_hp = enemy_hp
        
        # =====================================================================
        # ANTI-STUCK MECHANISM: Boss menu screen
        # Detect white box at (610,310) to (679,345) - Hornet's head on menu
        # If ALL pixels in this box are white for >2 seconds, press b
        # =====================================================================
        with mss() as sct:
            frame = np.asarray(sct.grab(self.monitor), dtype=np.uint8)
        
        # Check specific box region for all-white pixels
        check_box = frame[320:345, 610:679, :3]  # y1:y2, x1:x2
        all_white = (check_box > 240).all()  # All pixels > 240 in all channels
        
        if all_white:
            if self._menu_stuck_time is None:
                self._menu_stuck_time = time.time()
            elif (time.time() - self._menu_stuck_time) > 2.0:
                print("\n>>> ANTI-STUCK: Menu screen detected for >2s, pressing b...")
                pydirectinput.keyDown('b')
                time.sleep(0.1)
                pydirectinput.keyUp('b')
                self._menu_stuck_time = time.time()  # Reset timer
        else:
            self._menu_stuck_time = None  # Reset if not all white
        
        # =====================================================================
        # WIN/LOSE DETECTION
        # VICTORY: Boss HP was <5% once + HP bar gone >3s + Hornet HP > 0
        # LOSS: Hornet HP = 0 for >1.5 seconds
        # =====================================================================
        
        win = False
        lose = False
        
        # Track if boss HP ever dropped below 5% (needed for victory condition)
        # Use _lowest_enemy_hp which is the reliable tracked value
        if self._lowest_enemy_hp < 0.05:
            self._boss_hp_low_seen = True
        
        # Track HP bar visibility
        # BUT: If we got an impossible reading (white flash), that's NOT a real HP bar!
        real_hp_bar_visible = hp_bar_visible and not impossible_reading
        
        if real_hp_bar_visible:
            self._hp_bar_disappeared_time = None
            self._boss_hp_bar_seen = True
        else:
            # HP bar gone OR we're seeing false readings (white flash during victory)
            if self._boss_hp_bar_seen and self._hp_bar_disappeared_time is None:
                self._hp_bar_disappeared_time = time.time()
        
        # -----------------------------------------------------------------
        # BLACKSCREEN TRACKING (for defeat detection and punishment)
        # -----------------------------------------------------------------
        blackscreen_penalty = 0.0
        
        if is_blackscreen:
            if self._blackscreen_start_time is None:
                # Blackscreen just started
                self._blackscreen_start_time = time.time()
                # Punish ONCE per blackscreen session
                if not self._blackscreen_punished:
                    blackscreen_penalty = -0.5  # Punishment for entering blackscreen
                    self._blackscreen_punished = True
        else:
            # Blackscreen ended
            self._blackscreen_start_time = None
            self._blackscreen_punished = False  # Reset for next blackscreen session
        
        # -----------------------------------------------------------------
        # LOSS DETECTION: Blackscreen for more than 1.8 seconds
        # -----------------------------------------------------------------
        if is_blackscreen and self._blackscreen_start_time is not None:
            blackscreen_duration = time.time() - self._blackscreen_start_time
            if blackscreen_duration > 1.8:
                lose = True
                print(f"\n>>> DEFEATED! (Blackscreen for {blackscreen_duration:.1f}s)")
        
        # -----------------------------------------------------------------
        # VICTORY DETECTION: Boss HP was <5% once + HP bar gone >3s + alive
        # -----------------------------------------------------------------
        if (self._boss_hp_low_seen and 
            self._hp_bar_disappeared_time is not None and
            hornet_hp > 0):
            time_gone = time.time() - self._hp_bar_disappeared_time
            if time_gone > 3.0:
                win = True
                self._last_episode_was_victory = True
                print(f"\n>>> VICTORY! (Boss HP was <5%, HP bar gone {time_gone:.1f}s, Hornet alive)")
        
        done = win or lose
        
        # Calculate hurt/hit/heal
        hurt = hornet_hp < self.prev_hornet_hp
        hit = enemy_hp < self.prev_enemy_hp
        healed = hornet_hp > self.prev_hornet_hp  # Successful heal!
        
        # =================================================================
        # STEP REWARDS (clipped to [-2, 2])
        # =================================================================
        step_reward = action_rew
        
        # Survival reward: small bonus for every step alive (encourages staying alive)
        step_reward += 0.02  # ~1.2 reward per minute of survival
        
        # Blackscreen penalty (once per session)
        step_reward += blackscreen_penalty
        
        # Damage penalty (taking damage)
        if hurt:
            step_reward -= self.damage_penalty
            
            # COMBO DAMAGE PENALTY: If hit twice within 1.5 seconds, extra punishment
            current_time = time.time()
            if self._last_hurt_time is not None:
                time_since_last_hurt = current_time - self._last_hurt_time
                if time_since_last_hurt < 1.5:
                    step_reward -= 0.3  # Extra penalty for taking combo damage
            self._last_hurt_time = current_time
        
        # Hit reward (dealing damage to boss)
        if hit:
            step_reward += self.hit_reward
            # Extra bonus based on how much damage dealt (percentage decrease)
            damage_dealt = self.prev_enemy_hp - enemy_hp
            step_reward += damage_dealt * 2.0  # +0.02 per 1% damage (extra bonus)
            self._last_damage_time = time.time()  # Reset inactivity timer
        
        # Heal reward
        if healed:
            step_reward += self.heal_reward
        
        # Bonus for smart silk usage (spell hit) - DECREASED
        if used_spell and hit and self.prev_silk >= self.SILK_COST_SPELL:
            step_reward += 0.1  # Small bonus for landing a spell hit (was 0.2)
        
        # Encourage healing when low HP (attempt bonus)
        if used_heal and self.prev_silk >= self.SILK_COST_HEAL:
            if self.prev_hornet_hp <= 3:  # Critical HP
                step_reward += 0.3  # Strong bonus for clutch heal
            elif self.prev_hornet_hp <= 5:  # Low HP
                step_reward += 0.15  # Moderate bonus for smart heal
        
        # SUCCESSFUL HEAL BONUS: HP was < 7, and HP increased by 3
        if healed and self.prev_hornet_hp < 7:
            hp_gained = hornet_hp - self.prev_hornet_hp
            if hp_gained >= 3:
                step_reward += 1.5  # BIG reward for successful heal at low HP (increased)
        
        # Inactivity penalty (no damage dealt for too long)
        time_since_damage = time.time() - self._last_damage_time
        if time_since_damage > self.inactivity_window:
            # Apply penalty once per inactivity window
            windows_inactive = int(time_since_damage / self.inactivity_window)
            step_reward -= self.inactivity_penalty * min(windows_inactive, 3)  # Cap at 3x penalty
        
        # Movement inactivity penalty (not moving left/right for >2s)
        time_since_move = time.time() - self._last_move_time
        if time_since_move > 2.0:
            # Apply penalty similar to damage inactivity
            windows_no_move = int(time_since_move / 2.0)
            step_reward -= self.inactivity_penalty * min(windows_no_move, 3)  # Cap at 3x
        
        # Displacement inactivity penalty (not jumping/dashing for >5s)
        time_since_disp = time.time() - self._last_displacement_time
        if time_since_disp > 5.0:
            # Apply penalty similar to damage inactivity
            windows_no_disp = int(time_since_disp / 5.0)
            step_reward -= self.inactivity_penalty * min(windows_no_disp, 3)  # Cap at 3x
        
        # Clip step rewards only
        step_reward = np.clip(step_reward, -2.0, 2.0)
        
        # =================================================================
        # EPISODE REWARDS (NOT clipped - victory/defeat should be impactful)
        # =================================================================
        episode_reward = 0.0
        
        if win:
            # Victory: base + HP bonus + speed bonus
            # INCREASED: Base reward is now higher to make winning clearly better
            episode_time = time.time() - self._episode_time
            hp_bonus = (hornet_hp / 9.0) * 8.0  # +0 to +8 based on HP remaining (increased)
            speed_bonus = 90.0 / max(episode_time, 30.0)  # Faster = better (increased)
            episode_reward = self.victory_base + hp_bonus + speed_bonus
            print(f"\n>>> VICTORY! Reward: {episode_reward:.2f} (base:{self.victory_base} + hp:{hp_bonus:.2f} + speed:{speed_bonus:.2f})")
            print(f">>> Episode time: {episode_time:.1f}s, HP remaining: {hornet_hp}/9")
        elif lose:
            # Defeat: Scaled penalty based on boss HP
            # Above 50%: Heavy punishment
            # Below 50%: Reduced punishment (reward for progress)
            # Minimum penalty of -3 so death is ALWAYS bad
            MIN_DEFEAT_PENALTY = 3.0
            
            if enemy_hp >= 0.5:
                # Above 50%: Heavy punishment, scaled from -defeat_base to -(defeat_base + extra)
                # At 100%: -(defeat_base + 10), At 50%: -defeat_base
                extra_penalty = (enemy_hp - 0.5) * 20.0  # +0 to +10 extra penalty
                episode_reward = -(self.defeat_base + extra_penalty)
            else:
                # Below 50%: Reduced punishment (reward for getting boss HP low)
                # At 50%: -defeat_base, At 0%: -MIN_DEFEAT_PENALTY
                # Linear interpolation between defeat_base and MIN_DEFEAT_PENALTY
                progress_factor = enemy_hp / 0.5  # 1.0 at 50%, 0.0 at 0%
                penalty = MIN_DEFEAT_PENALTY + (self.defeat_base - MIN_DEFEAT_PENALTY) * progress_factor
                episode_reward = -penalty
            
            print(f"\n>>> DEFEATED! Penalty: {episode_reward:.2f} (boss_hp:{enemy_hp*100:.1f}%)")
            if enemy_hp < 0.5:
                print(f">>> Progress bonus applied! (below 50%)")
        
        # Final reward = step reward + episode reward
        reward = step_reward + episode_reward
        
        # -----------------------------------------------------------------
        # TIMEOUT DETECTION: Episode stuck for more than 5 minutes
        # (Must check BEFORE cleanup which sets _episode_time to None)
        # -----------------------------------------------------------------
        truncated = False
        if not done and self._episode_time is not None:
            episode_duration = time.time() - self._episode_time
            
            if episode_duration > 300:  # 5 minutes = 300 seconds
                print(f"\n>>> TIMEOUT! Episode stuck for {episode_duration:.0f}s (>5 min)")
                print(">>> Ending episode. reset() will handle the restart.")
                
                done = True
                truncated = True
                reward = 0  # Neutral reward for timeout (not agent's fault)
                self._timeout_occurred = True  # Flag for reset() to skip long wait
        
        # Cleanup or update state
        if done:
            self.cleanup()
        else:
            self.prev_hornet_hp = hornet_hp
            self.prev_enemy_hp = enemy_hp
            self.prev_silk = silk_amount
        
        # Include silk amount in info for debugging/monitoring
        info = {
            'hornet_hp': hornet_hp,
            'enemy_hp': enemy_hp,
            'silk': silk_amount,
            'timeout': truncated  # Flag to indicate this episode was a timeout
        }
        return obs, reward, done, truncated, info

    # =========================================================================
    # RESET FUNCTION
    # =========================================================================
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.
        
        After defeating Lace (or dying), navigate back to restart the fight:
        1. Wait for post-fight sequence (~6 seconds)
        2. Press b + space + space to restart
        
        :return: (initial observation, info dict)
        """
        super(LaceEnv, self).reset(seed=seed)
        self.cleanup()
        
        if self._first_reset:
            # First reset: user is already in the fight, just start!
            print("\n" + "=" * 40)
            print("RESET: First episode - assuming already in fight")
            print("RESET: Starting in 2 seconds...")
            time.sleep(2.0)
            print("RESET: GO!")
            print("=" * 40 + "\n")
            self._first_reset = False
        else:
            # Subsequent resets: navigate back to fight
            print("\n" + "=" * 40)
            
            # Determine wait time based on how episode ended
            if self._timeout_occurred:
                # Timeout: minimal wait, just need to get to menu
                wait_time = 1.0
                print(f"RESET: Timeout recovery - minimal wait ({wait_time:.0f}s)...")
                self._timeout_occurred = False  # Reset flag
            elif self._last_episode_was_victory:
                # Victory animation is longer
                wait_time = 10.0
                print(f"RESET: Victory! Waiting {wait_time:.0f} seconds...")
            else:
                # Normal defeat
                wait_time = 4.0
                print(f"RESET: Defeat. Waiting {wait_time:.0f} seconds...")
            
            time.sleep(wait_time)
            self._last_episode_was_victory = False  # Reset for next episode
            
            # Navigate to restart the fight: b -> space -> space
            print("RESET: Pressing b...")
            pydirectinput.keyDown('b')
            time.sleep(0.1)
            pydirectinput.keyUp('b')
            time.sleep(0.2)
            
            print("RESET: Pressing space...")
            pydirectinput.keyDown('space')
            time.sleep(0.1)
            pydirectinput.keyUp('space')
            time.sleep(0.2)
            
            print("RESET: Pressing space again...")
            pydirectinput.keyDown('space')
            time.sleep(0.1)
            pydirectinput.keyUp('space')
            
            # Wait for loading/transition to finish
            print("RESET: Waiting 4 seconds...")
            time.sleep(4.0)
            
            # Navigate into arena: d + l (dash right)
            print("RESET: Pressing d + l (dash right)...")
            pydirectinput.keyDown('d')
            time.sleep(0.05)
            pydirectinput.keyDown('l')
            time.sleep(0.1)
            pydirectinput.keyUp('l')
            pydirectinput.keyUp('d')
            
            print("RESET: Waiting 1 second...")
            time.sleep(1.0)
            print("RESET: Done! Starting new episode.")
            print("=" * 40 + "\n")
        
        self.prepare()
        return self.observe()[0], {}

    def prepare(self):
        """Initialize HP and silk tracking for new episode."""
        self.prev_hornet_hp = len(self.HP_CKPT)
        self.prev_enemy_hp = 1.0
        self.prev_silk = 18  # Assume full silk at start (18 units max)
        self._hp_bar_disappeared_time = None  # Reset HP bar tracking
        self._boss_hp_bar_seen = False  # Track if HP bar was ever visible this episode
        self._boss_hp_low_seen = False  # Track if boss HP ever dropped below 5%
        self._tool_uses = 0  # Reset tool counter
        self._last_tool_time = None  # Reset tool cooldown
        self._hornet_zero_hp_time = None  # Reset anti-stuck timer
        self._menu_stuck_time = None  # Reset menu anti-stuck timer
        self._last_damage_time = time.time()  # Track inactivity (start fresh)
        self._last_move_time = time.time()  # Track movement inactivity
        self._last_displacement_time = time.time()  # Track displacement inactivity
        self._lowest_enemy_hp = 1.0  # Reset lowest boss HP tracker
        self._blackscreen_start_time = None  # Reset blackscreen tracking
        self._blackscreen_punished = False  # Reset blackscreen punishment flag
        self._last_hurt_time = None  # Reset combo damage tracking
        self._episode_time = time.time()
        self._prev_time = time.time()

    def close(self):
        """Clean up when environment is closed."""
        self.cleanup()

    def cleanup(self):
        """
        Do any necessary cleanup on the interaction.
        Should only be called before or after an episode.
        """
        if self._timer is not None:
            self._timer.join()
        self.holding = []
        
        for key in self.KEYMAPS.values():
            if isinstance(key, tuple):
                for k in key:
                    pydirectinput.keyUp(k)
            else:
                pydirectinput.keyUp(key)
        
        self.prev_hornet_hp = None
        self.prev_enemy_hp = None
        self.prev_silk = None
        self.prev_action = -1
        self._hp_bar_disappeared_time = None
        self._boss_hp_bar_seen = False
        self._boss_hp_low_seen = False
        self._hornet_zero_hp_time = None
        self._menu_stuck_time = None
        self._last_tool_time = None  # Reset tool cooldown
        self._timer = None
        self._episode_time = None
        self._prev_time = None
        gc.collect()


# =============================================================================
# ALTERNATIVE ENVIRONMENT VERSION (with different hyperparameters)
# =============================================================================

class LaceEnvV2(LaceEnv):
    """
    Alternative version with tuned hyperparameters.
    """
    
    REWMAPS = {
        Move.HOLD_LEFT: 0,
        Move.HOLD_RIGHT: 0,
        Displacement.TIMED_SHORT_JUMP: 0,
        Displacement.TIMED_LONG_JUMP: 0,
        Displacement.DASH: 0,
        Attack.ATTACK: -2e-6,   # Small penalty for spamming attack
        Attack.SPELL: -2e-6,    # Small penalty for spamming spell
        Attack.TOOL: -2e-6,     # Small penalty for spamming tool
        Heal.HEAL: 0,           # No penalty for heal attempt
    }

    def __init__(self, obs_shape=(192, 192), rgb=False, gap=0.17,
                 damage_penalty=0.5, hit_reward=0.85, heal_reward=0.7,
                 inactivity_penalty=0.3, inactivity_window=5.0,
                 victory_base=10.0, defeat_base=8.0):
        super().__init__(obs_shape, rgb, gap, damage_penalty, hit_reward, heal_reward,
                         inactivity_penalty, inactivity_window, victory_base, defeat_base)
        self._hold_time = self.gap * 0.97
        self._fail_hold_rew = -1e-5


# =============================================================================
# UTILITY FUNCTIONS FOR CALIBRATION
# =============================================================================

def test_window_detection():
    """Test if the environment can find the Silksong window."""
    print("Testing window detection...")
    try:
        env = LaceEnv()
        print(f"Window found! Monitor region: {env.monitor}")
        env.close()
        return True
    except Exception as e:
        print(f"Failed to find window: {e}")
        return False


def test_screenshot():
    """Take a test screenshot and save it for calibration."""
    print("Taking test screenshot...")
    try:
        env = LaceEnv()
        obs, hornet_hp, enemy_hp, silk, hp_bar_visible, is_blackscreen = env.observe()
        
        print(f"Observation shape: {obs.shape}")
        print(f"Hornet HP: {hornet_hp}")
        print(f"Enemy HP: {enemy_hp:.2f}")
        print(f"Silk: {silk}")
        print(f"HP Bar Visible: {hp_bar_visible}")
        print(f"Is Blackscreen: {is_blackscreen}")
        
        # Save raw screenshot
        with mss() as sct:
            frame = np.asarray(sct.grab(env.monitor), dtype=np.uint8)
        
        cv2.imwrite('test_screenshot_raw.png', frame)
        print("Saved: test_screenshot_raw.png")
        
        # Save processed observation
        obs_display = obs[0] if len(obs.shape) == 3 else obs
        cv2.imwrite('test_screenshot_obs.png', obs_display)
        print("Saved: test_screenshot_obs.png")
        
        env.close()
        return True
    except Exception as e:
        print(f"Screenshot failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def calibrate_hp_positions():
    """
    Helper function to calibrate HP bar positions.
    Run this to take a screenshot and manually identify HP positions.
    """
    print("=" * 60)
    print("HP CALIBRATION HELPER")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Run the game in 1280x720 windowed mode")
    print("2. Enter the Lace arena")
    print("3. Pause the game or wait for a clear frame")
    print("4. Run this function")
    print("5. Open 'calibration_screenshot.png' in an image editor")
    print("6. Find the Y-coordinate of the health bar row")
    print("7. Find the X-coordinates of each health bead center")
    print("8. Update HP_BAR_Y and HP_CKPT in LaceEnv")
    print("=" * 60)
    
    try:
        env = LaceEnv()
        with mss() as sct:
            frame = np.asarray(sct.grab(env.monitor), dtype=np.uint8)
        
        cv2.imwrite('calibration_screenshot.png', frame)
        print(f"\nSaved: calibration_screenshot.png")
        print(f"Image size: {frame.shape[1]}x{frame.shape[0]}")
        
        env.close()
    except Exception as e:
        print(f"Calibration failed: {e}")
        import traceback
        traceback.print_exc()


def print_action_space():
    """Print the action space for reference."""
    print("=" * 60)
    print("LACE ENVIRONMENT ACTION SPACE")
    print("=" * 60)
    print(f"\nMove actions ({len(Move)} options):")
    for m in Move:
        print(f"  {m.value}: {m.name}")
    
    print(f"\nAttack actions ({len(Attack)} options):")
    for a in Attack:
        key = LaceEnv.KEYMAPS.get(a, 'N/A')
        print(f"  {a.value}: {a.name} -> {key}")
    
    print(f"\nDisplacement actions ({len(Displacement)} options):")
    for d in Displacement:
        key = LaceEnv.KEYMAPS.get(d, 'N/A')
        print(f"  {d.value}: {d.name} -> {key}")
    
    print(f"\nHeal actions ({len(Heal)} options):")
    for h in Heal:
        key = LaceEnv.KEYMAPS.get(h, 'N/A')
        print(f"  {h.value}: {h.name} -> {key}")
    
    total = len(Move) * len(Attack) * len(Displacement) * len(Heal)
    print(f"\nTotal discrete actions: {total}")
    print("(Move × Attack × Displacement × Heal)")
    print("=" * 60)


def debug_hp_detection():
    """Debug function to see what HP values the code detects."""
    print("=" * 60)
    print("HP DETECTION DEBUG")
    print("=" * 60)
    print("\nSwitch to the game window now!")
    
    # Countdown
    for i in range(5, 0, -1):
        print(f"  Starting in {i}...")
        time.sleep(1)
    print("  Capturing!\n")
    
    try:
        env = LaceEnv()
        with mss() as sct:
            frame = np.asarray(sct.grab(env.monitor), dtype=np.uint8)
        
        # Hornet HP detection
        print("\n--- HORNET HP ---")
        print(f"HP_BAR_Y = {env.HP_BAR_Y}")
        print(f"HP_CKPT = {env.HP_CKPT}")
        
        hornet_hp_bar = frame[env.HP_BAR_Y, :, 0]
        for i, x in enumerate(env.HP_CKPT):
            pixel_val = hornet_hp_bar[x]
            status = "FULL" if pixel_val > 200 else "EMPTY"
            print(f"  Bead {i+1} at X={x}: pixel={pixel_val} -> {status}")
        
        checkpoint1 = hornet_hp_bar[env.HP_CKPT]
        checkpoint2 = hornet_hp_bar[env.HP_CKPT - 1]
        hornet_hp = ((checkpoint1 > 200) | (checkpoint2 > 200)).sum()
        print(f"\n  Detected Hornet HP: {hornet_hp}/{len(env.HP_CKPT)}")
        
        # Silk detection
        print("\n--- SILK METER (18 units) ---")
        print(f"SILK_BAR_Y = {env.SILK_BAR_Y}")
        print(f"SILK_CKPT = {env.SILK_CKPT}")
        print(f"SILK_THRESHOLD = {env.SILK_THRESHOLD}")
        
        silk_bar = frame[env.SILK_BAR_Y, :, 0]
        print(f"\nSilk units (left to right):")
        for i, x in enumerate(env.SILK_CKPT):
            pixel_val = silk_bar[x]
            status = "FULL" if pixel_val > env.SILK_THRESHOLD else "EMPTY"
            print(f"  Unit {i+1:2d} at X={x:3d}: pixel={pixel_val:3d} -> {status}")
        
        silk_values = silk_bar[env.SILK_CKPT]
        silk_amount = (silk_values > env.SILK_THRESHOLD).sum()
        print(f"\n  Detected Silk: {silk_amount}/18")
        
        # Enemy HP detection
        print("\n--- ENEMY (LACE) HP ---")
        enemy_hp_bar_y = 680
        enemy_hp_bar_x_start = 348
        enemy_hp_bar_x_end = 947
        
        # NOTE: x_end is INCLUSIVE, so +1 for Python slicing
        enemy_hp_bar = frame[enemy_hp_bar_y, enemy_hp_bar_x_start:enemy_hp_bar_x_end + 1, :]
        
        print(f"HP bar at Y={enemy_hp_bar_y}, X={enemy_hp_bar_x_start}-{enemy_hp_bar_x_end} (inclusive)")
        print(f"Bar length: {len(enemy_hp_bar)} pixels")
        
        # Expected colors:
        # Health: #beb8b8ff → B=190, G=184, R=184 (grayish)
        # Empty:  #ffffffff → B=255, G=255, R=255 (white)
        print(f"\nExpected colors:")
        print(f"  Health: B~190, G~184, R~184 (grayish)")
        print(f"  Empty:  B~255, G~255, R~255 (white)")
        
        print(f"\nSample pixels from LEFT side (health remaining):")
        for i in range(min(5, len(enemy_hp_bar))):
            b, g, r, a = enemy_hp_bar[i]
            if 170 <= b <= 220:
                status = "HEALTH (gray)"
            elif b > 240:
                status = "EMPTY (white)"
            else:
                status = "UNKNOWN"
            print(f"  X={enemy_hp_bar_x_start + i}: B={b}, G={g}, R={r} -> {status}")
        
        print(f"\nSample pixels from RIGHT side (possibly missing health):")
        for i in range(max(0, len(enemy_hp_bar)-5), len(enemy_hp_bar)):
            b, g, r, a = enemy_hp_bar[i]
            if 170 <= b <= 220:
                status = "HEALTH (gray)"
            elif b > 240:
                status = "EMPTY (white)"
            else:
                status = "UNKNOWN"
            print(f"  X={enemy_hp_bar_x_start + i}: B={b}, G={g}, R={r} -> {status}")
        
        # Count using new logic
        b_channel = enemy_hp_bar[..., 0]
        health_pixels = ((b_channel >= 170) & (b_channel <= 220)).sum()
        empty_pixels = (b_channel > 240).sum()
        unknown_pixels = len(enemy_hp_bar) - health_pixels - empty_pixels
        
        recognized = health_pixels + empty_pixels
        if recognized > 0:
            enemy_hp = health_pixels / recognized
        else:
            enemy_hp = 0
        
        print(f"\nHealth pixels (B=170-220): {health_pixels}")
        print(f"Empty pixels (B>240): {empty_pixels}")
        print(f"Unknown pixels: {unknown_pixels}")
        print(f"Bar visible: {recognized >= len(enemy_hp_bar) * 0.8}")
        print(f"Enemy HP: {enemy_hp:.1%} ({enemy_hp:.2f})")
        
        env.close()
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"Debug failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            test_window_detection()
        elif sys.argv[1] == 'screenshot':
            test_screenshot()
        elif sys.argv[1] == 'calibrate':
            calibrate_hp_positions()
        elif sys.argv[1] == 'actions':
            print_action_space()
        elif sys.argv[1] == 'debug':
            debug_hp_detection()
    else:
        print("Usage:")
        print("  python lace_env.py test       - Test window detection")
        print("  python lace_env.py screenshot - Take test screenshot")
        print("  python lace_env.py calibrate  - HP calibration helper")
        print("  python lace_env.py actions    - Print action space info")
        print("  python lace_env.py debug      - Debug HP detection values")


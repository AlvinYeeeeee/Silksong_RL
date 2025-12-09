# Silksong RL Changelog

## [2024-12-03] Tool Cooldown & Reward Adjustments

### Changes Made

#### 1. Tool Cooldown System
- **5 second cooldown** between tool uses
- If tool used during cooldown: **-0.3 penalty**, action replaced with no-op
- Tool limit remains at **8 uses per episode**
- Cooldown tracked via `_last_tool_time` variable

#### 2. Survival Reward
- Added **+0.02 per step** for staying alive
- Encourages survival (~1.2 reward per minute of play)

#### 3. Increased Boss Damage Reward
- Base `hit_reward`: 0.7 → **0.85**
- **Extra damage bonus**: +2.0 * (percentage HP lost)
- Example: 2% boss HP drop = +0.04 extra bonus

#### 4. Increased Heal Reward
- Base `heal_reward`: 0.5 → **0.7**
- Successful heal bonus (HP < 7, gained 3+): 1.2 → **1.5**

### New Reward Summary

| Action | Reward |
|--------|--------|
| Survival (per step) | +0.02 |
| Hit boss | +0.85 + (damage% * 2.0) |
| Take damage | -0.35 |
| Heal success | +0.7 |
| Clutch heal (HP ≤ 3) | +0.3 bonus |
| Low HP heal (HP ≤ 5) | +0.15 bonus |
| Successful heal (HP < 7, +3 HP) | +1.5 bonus |
| Tool during cooldown | -0.3 |
| Tool when depleted | -0.5 |

### Rationale
- Tool cooldown prevents spamming (5s matches game's internal cooldown feel)
- Survival reward encourages staying alive longer
- Higher hit rewards incentivize aggression and boss engagement
- Higher heal rewards make healing more attractive vs. attacking

### Files Modified
- `lace_env.py`: Tool cooldown logic, survival reward, increased hit/heal rewards
- `train.py`: Updated default parameters

### Migration Note
These changes are mostly reward adjustments. Explorations **should be re-run** for best results, but existing explorations may still work reasonably.

---

## [2024-12-03] Best HP Model Tracking

### Added: "besthp" Model Checkpoint

New model checkpoint that saves the model dealing the **most damage to the boss**.

#### How It Works
- Every 10 episodes (during evaluation), boss HP at episode end is tracked
- Average boss HP over the last 10 evaluations is calculated
- When average boss HP is **lower than previous best**, model is saved as `besthp`
- Requires at least 3 evaluations before saving (to avoid noise)

#### Model Checkpoints Summary

| Model | Saved When | What It Optimizes |
|-------|------------|-------------------|
| `best` | Highest evaluation reward | Overall performance (reward) |
| `besthp` | Lowest average boss HP | Most damage dealt to boss |
| `besttrain` | Highest training reward | Best training episode |
| `latest` | Every episode | Most recent model |
| `final` | End of training | Final trained model |

#### Rationale
- `best` might favor survival over damage
- `besthp` directly tracks your goal: getting boss HP as low as possible
- Use `besthp` if you want the most aggressive, damage-dealing model

#### Files Modified
- `train.py`: Added `evaluate_with_hp()` function, boss HP tracking, `besthp` saving logic

---

## [2024-12-01] Reward System Overhaul

### Problem Observed
After initial training, the agent learned to **hide in the top-left corner** and avoid engaging with the boss entirely. This happened because:
- Surviving (not taking damage) gave a consistent positive reward stream
- The idle penalty was too tiny (-0.00008 per step) to matter
- The defeat penalty (~-0.2) was weaker than the damage penalty (-0.8)
- The agent discovered that **hiding > fighting** since hiding had guaranteed returns

### Changes Made

#### Removed Parameters
| Parameter | Old Value | Purpose |
|-----------|-----------|---------|
| `w1` | 0.8 | Damage penalty weight |
| `w2` | 0.5 | Hit reward weight |
| `w3` | -8e-5 | Idle penalty (per step) |

#### New Parameters
| Parameter | Value | Purpose |
|-----------|-------|---------|
| `damage_penalty` | 0.5 | Penalty when Hornet takes damage |
| `hit_reward` | 0.7 | Reward for hitting boss |
| `heal_reward` | 0.5 | Reward for successful heal |
| `inactivity_penalty` | 0.3 | Penalty per inactivity window of not dealing damage |
| `inactivity_window` | 5.0 | Seconds before inactivity penalty kicks in |
| `victory_base` | 10.0 | Base victory reward (NOT clipped) |
| `defeat_base` | 8.0 | Base defeat penalty (NOT clipped) |

#### New Reward Structure

**Step Rewards (clipped to [-2, 2]):**
- Hit boss: +0.7
- Take damage: -0.5
- Heal success: +0.5
- Land spell hit: +0.2 bonus
- Clutch heal (HP ≤ 3): +0.3 bonus
- Low HP heal (HP ≤ 5): +0.15 bonus
- Inactivity (>5s no damage): -0.3 per window (capped at 3x)

**Episode Rewards (NOT clipped):**
- **Victory**: `+10.0 + (hornet_hp/9)*5 + 60/time`
  - Base: +10
  - HP bonus: +0 to +5 based on remaining HP
  - Speed bonus: faster wins = more reward
  - Total range: ~+10 to +17

- **Defeat**: `-(8.0 + enemy_hp*7)`
  - Base: -8
  - Boss HP penalty: -0 to -7 based on how much boss HP left
  - Total range: ~-8 to -15
  - Dying with boss at 100% HP = -15 (catastrophic)
  - Dying with boss at 10% HP = -8.7 (still bad, but less so)

#### Clipping Strategy
- **Step rewards**: Clipped to [-2, 2] to prevent single steps from dominating
- **Episode rewards**: NOT clipped - victory/defeat should have BIG impact

### Rationale
1. **Heavy defeat penalty**: Agent must fear death more than fear of engaging
2. **Scaled by boss HP**: Dying after dealing damage is less bad than dying immediately
3. **Inactivity punishment**: Forces engagement, prevents corner camping
4. **Victory celebration**: Big positive signal for winning, bonus for speed
5. **Reduced damage penalty**: Encourage risk-taking (0.5 vs old 0.8)

### Files Modified
- `lace_env.py`: Reward calculation, parameters, inactivity tracking
- `train.py`: Environment initialization with new parameters

### Migration Note
**IMPORTANT**: Since reward function changed significantly, **exploration data must be re-collected**. Old exploration files have rewards calculated with the old formula and will give inconsistent signals.

---

## [2024-12-02] Major Update - Detection & Timeout

### Changes Made

#### 1. Blackscreen Handling
- **HP/Silk Preservation**: During blackscreen, keep previous HP and silk values (don't read 0)
- **Blackscreen Penalty**: -0.5 penalty ONCE per blackscreen session (teaches to avoid lava)
- **Defeat Detection**: Changed from "HP=0 for 1.5s" to "Blackscreen for 1.8s"

#### 2. Timeout Mechanism
- If episode runs > 5 minutes, force end
- Neutral reward (0) for timeout
- Minimal wait (1s) before reset
- Next episode starts normally

#### 3. Reset Wait Times
| Episode End | Wait Before Reset |
|-------------|-------------------|
| Timeout | 1 second |
| Victory | 10 seconds |
| Defeat | 4 seconds |

### Migration Note
**⚠️ REDO EXPLORATIONS REQUIRED**
- Defeat detection changed (blackscreen vs HP=0)
- Reward values changed significantly
- New blackscreen penalty added

```bash
rm -r ./explorations/*
python train.py
```

---

## [2024-12-02] Reward Rebalance - Victory/Defeat

### Changes Made

#### Victory Reward (INCREASED)
- Base: 10 → **18**
- HP bonus: 5 → **8** (scales with remaining HP)
- Speed bonus: 60 → **90** (faster = more reward)
- **Total range: ~+18 to +29**

#### Defeat Penalty (REDESIGNED)
New formula with progress recognition:

| Boss HP | Defeat Penalty |
|---------|----------------|
| 100% | -20 (very bad) |
| 75% | -15 |
| 50% | -10 (base) |
| 25% | -6.5 |
| 10% | -4.4 |
| 0% | -3 (minimum, death always bad) |

**Key insight**: Getting boss below 50% reduces punishment, rewarding progress even in death.

#### Rationale
- Winning should feel GREAT (+18 to +29)
- Dying at 100% boss HP should feel TERRIBLE (-20)
- Dying at 10% boss HP should feel "almost had it" (-4.4)
- Death is ALWAYS punished (minimum -3)

---

## [2024-12-02] Movement & Healing Incentives

### Problem Observed
After training, the bot learned to **stand still and spam attack** without moving or dodging. This works because boss damage isn't lethal quickly, so it can deal damage while tanking hits.

### Changes Made

#### 1. Movement Inactivity Penalty
If not moving left/right for >2 seconds, apply penalty (`inactivity_penalty` per 2s window, capped at 3x).

#### 2. Displacement Inactivity Penalty
If not jumping/dashing for >5 seconds, apply penalty (`inactivity_penalty` per 5s window, capped at 3x).

#### 3. Successful Heal Bonus
If HP was < 7 and HP increased by ≥3 (successful heal), reward +0.8.

### New Reward Structure

| Condition | Penalty/Reward |
|-----------|----------------|
| No movement for >2s | -0.3 per 2s window (max -0.9) |
| No jump/dash for >5s | -0.3 per 5s window (max -0.9) |
| Successful heal (HP<7, gained 3+) | +0.8 |

### Rationale
- Forces the bot to **actively move** to avoid attacks
- Encourages **dodging** with jumps and dashes
- Strongly rewards **smart healing** when low HP

---

## [2024-12-02] Boss HP Detection Fixes

### Problems Observed
1. **Boss HP "jumping" during victory animation**: White flashes caused false HP readings (2.5% → 100%)
2. **HP bar visibility detection failing at very low HP**: When boss had ~2% HP, variance dropped and bar was falsely marked as "missing"
3. **Victory not triggering**: White flashes kept resetting the "HP bar disappeared" timer

### Fixes Applied

#### 1. Boss HP Sanity Check
Boss HP can **NEVER increase**. If we detect an increase, it's a false reading.

```python
# Track lowest HP seen (can only go DOWN)
_lowest_enemy_hp = 1.0

# If raw_enemy_hp > lowest + 1%, it's impossible - use lowest
if raw_enemy_hp > _lowest_enemy_hp + 0.01:
    enemy_hp = _lowest_enemy_hp  # Keep the reliable value
```

#### 2. Color-Based HP Bar Detection
Instead of variance-based detection, use exact game colors:
- **Health color**: `#beb8b8ff` → B channel 170-220
- **Empty color**: `#ffffffff` → B channel > 240
- Bar visible if 80%+ pixels match expected colors

#### 3. Fixed HP Bar Disappearance Tracking
Impossible readings (white flashes) should NOT reset the victory timer:

```python
real_hp_bar_visible = hp_bar_visible and not impossible_reading
```

#### 4. Inclusive Pixel Range
Fixed Python slicing: `x_start:x_end` → `x_start:x_end+1` to include boundary pixels.

#### 5. Victory Animation Wait Time
- **Victory**: Wait 10 seconds before reset (longer animation)
- **Defeat**: Wait 4 seconds before reset

### Files Modified
- `lace_env.py`: HP detection, sanity check, reset timing
- `test_detection.py`: Created for manual testing

---

## [Previous Changes]
- Victory/Defeat detection updated to use HP-based conditions
- Wait time after defeat increased to 4 seconds
- Various key mapping and input fixes


# Lewis Signaling Game

A multi-agent simulation of emergent communication learning through reinforcement learning, inspired by evolutionary game theory and artificial life.


<div align="center">
  <img src="https://github.com/NullLabTests/lewis-signaling-game/blob/main/ss.png" alt="Screenshot of Streamlit UI" width="800" />
</div>

## What is a Lewis Signaling Game?

A **Lewis signaling game** is a game-theoretic model of communication between two agents: a **sender** and a **receiver**. Named after philosopher David K. Lewis, the game works as follows:

1. **Sender** observes a hidden state from a finite set (e.g., "red", "blue", "green")
2. **Sender** produces a signal (e.g., a sound, gesture, or chemical marker) chosen from a limited signal space
3. **Receiver** observes the signal (but not the true state) and performs an action
4. **Both agents reward** if the action matches the true state

### The Learning Problem

Neither agent has pre-coded knowledge of signal meanings. Instead, through repeated gameplay and reinforcement learning (REINFORCE), agents **discover optimal signaling conventions**—a shared language that emerges from scratch.

In our implementation:
- **Sender's policy**: P(signal | state) — learns to send consistent signals for each state
- **Receiver's policy**: P(action | signal) — learns to map signals to correct actions
- **Update rule**: Increase probabilities of signal/action pairs when reward=1, decrease on reward=0

**Result**: A minimal artificial language where meaning is grounded in successful coordination.

---

## Connection to Artificial Life (ALife)

Lewis signaling games are foundational in ALife because they model **evolutionary emergence of communication** without top-down design:

### Key ALife Principles

| Principle | How Lewis Game Models It |
|-----------|------------------------|
| Self-organization | Agents discover communication structure spontaneously via local updates |
| Emergence | Complex coordinated behavior (shared language) arises from simple rules |
| Adaptation | Policies evolve under selection pressure (reward gradient) |
| Information transfer | Communication substrate (signals) carries no inherent meaning—it is learned |
| Minimal substrate | No central controller; only local agent-agent interaction |

### Why It Matters for ALife

Classic ALife explores how life-like properties (replication, adaptation, evolution) emerge from simple rules. Communication is a cornerstone: in nature, molecular signaling cascades, ant pheromones, and bird songs all evolved without a "designer." Lewis games provide a **mathematical framework** to study this emergence.

---

## Molecular Language: Signaling via Macromolecules

Nature demonstrates the Lewis signaling game in **molecular biology**—perhaps the most elegant instance of emergent communication.

### How Molecules "Speak"

Cells don't have brains or words. Instead, they use **macromolecules** (proteins, nucleic acids, lipids) as a chemical language:

#### 1. Signal Molecules (Ligands)
- Hormones, neurotransmitters, cytokines
- Act as "utterances" sent by a sender cell
- Example: Epinephrine (adrenaline) is a signal molecule—a chemical message meaning "stress detected"

#### 2. Signal Receivers (Receptors)
- Proteins on cell membranes or inside cells
- Act as "listeners" that bind specific signals
- Example: β-adrenergic receptor binds epinephrine → initiates fight-or-flight response

#### 3. Meaning Through Binding
- A ligand's **meaning is not in its structure**—it emerges from:
  - Which receptor it binds (specificity)
  - What action the cell performs upon binding (consequence)
- **This is exactly the Lewis game**: signal (ligand) + receiver response (action) = coordinated state interpretation

#### 4. Protein Macromolecules as Language
Proteins are the **alphabet**:
- **Antibodies** recognize pathogens (signal = antigen structure → action = immune response)
- **Enzymes** recognize substrates (signal = molecular shape → action = chemical transformation)
- **Transcription factors** recognize DNA sequences (signal = DNA binding site → action = gene activation)

### A Concrete Example: Glucose Sensing

This is a **Lewis game in your cells right now**:

| Component | Game Analogy |
|-----------|--------------|
| State | Blood glucose level (high or low) |
| Sender | Pancreatic β-cell detects glucose |
| Signal | Releases insulin (macromolecule) |
| Receiver | Muscle/fat cell with insulin receptor |
| Action | Glucose uptake (coordinated response) |
| Reward | Successful homeostasis (stable glucose) |

Insulin is a **signal macromolecule**. Its "meaning" is **not intrinsic**—it emerges from the receptor's response. A cell lacking insulin receptors "doesn't speak insulin" and cannot understand the message.

### Evolution of Molecular Language

Like Lewis games, molecular signaling systems evolved without design:
1. **Primordial World**: Random proteins + random binding → occasional useful interactions
2. **Selection**: Proteins whose bindings improved survival replicated
3. **Elaboration**: Binding specificity refined; new ligand-receptor pairs co-evolved
4. **Standardization**: Successful pairs became conserved across species (e.g., insulin evolved 2.5 billion years ago, still used today)

**This is emergent communication through macromolecules**—exactly what our Lewis game models, but running on biological computers for eons.

## Demo Video

Watch the agents invent a shared language in real time:

[▶️ Play Demo Video](./demo.mp4)

### Modern Applications: Drug Design as "Language Learning"

Drug discovery often involves:
- Reverse-engineering the "meaning" of a signal (e.g., what does TNF-α signaling mean?)
- Designing new ligands (synthetic "words") that exploit or block that meaning
- Training a system to recognize and act on these new signals

AI-driven drug discovery (AlphaFold, graph neural nets) is learning the **molecular language**—teaching machines to predict how a macromolecule's structure encodes meaning through binding.

---
## Implementation

This repo implements a minimal Lewis game with pure NumPy:

```python
from src.game import LewisGame

game = LewisGame(n_states=3, n_signals=3, lr=0.2, seed=42)

for _ in range(2000):
    state = game.rng.integers(3)
    signal, action, reward = game.step(state)
    game.update(state, signal, action, reward)
```
## 🎥 Demo Video

**Watch the agents invent a shared language in real time (Streamlit dashboard):**

<video src="https://github.com/NullLabTests/lewis-signaling-game/raw/main/demo.mp4" 
       controls 
       width="100%" 
       style="max-width:800px; display:block; margin:0 auto; border-radius:8px;">
  Your browser does not support the video tag.
</video>

*(Click the play button — no download needed)*

## 🎥 Demo Video

**Watch the agents invent a shared language in real time:**

<video src="https://github.com/NullLabTests/lewis-signaling-game/raw/main/demo.mp4" 
       controls 
       width="100%" 
       style="max-width: 800px; border-radius: 8px; display: block; margin: 0 auto;">
  Your browser does not support the video tag.
</video>

*Playable inline — no download needed*

# CLAUDE.md

----
**MAIN RULE FOR CLAUDE**: 
Every writing should have the high signal/noise ratio and minimalism at a guiding principle. 
Less is More
Information density is the main metric to optimize
Bluff is to be minimized at all costs. 



Python AI library: unified interface for audio, text, vision tasks.

## Philosophy
AI models = learned programs that perform tasks.

**Core Principles:**
- Task-first UX: Users solve problems, not choose models
- Two-tier performance: `high_performance=False/True` 
- Progressive disclosure: Simple → Expert control
- Composable prompts: Extend or replace modules

**Architecture:**
- 2 models per task maximum
- Separate engines: Audio/Text/Vision
- Task verbs abstract model complexity

**Usage:**
```python
surus.transcribe(audio)                    # Default
surus.transcribe(audio, high_performance=True)  # Best quality
```

**Development:**
Build iteratively: MVP → Performance → Customization → Expert controls
# SURUS Design Rationale

## Task-First vs Model-First

**Model-First Problems:**
- Users choose models before understanding tasks
- Prompt engineering required
- Code breaks when models change

**SURUS Advantage:**
```python
# Model-first (current state)
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Transcribe this audio..."}]
)

# Task-first (SURUS approach)
surus.transcribe(audio_file)  # Just solve the task
```

**Benefits:**
- Users think tasks, not models
- Model improvements transparent
- No prompt engineering
- Focus on business logic

## Two-Tier Performance

**Rejected Alternatives:**
- Single model: Doesn't fit all budgets
- Model marketplace: Decision paralysis

**SURUS Solution:**
   ```python
   surus.transcribe(audio)                    # Good, cheap
   surus.transcribe(audio, high_performance=True)  # Best, expensive
   ```

**Why This Works:**
- One decision: quality vs cost
- 2 models per task maximum
- Room for later complexity

## Progressive Disclosure

**Layer 1: Zero-Config**
```python
result = surus.extract_to_json(document)
```
80% of use cases

**Layer 2: Performance**
```python
result = surus.extract_to_json(document, high_performance=True)
```
Better results when needed

**Layer 3: Customization**
```python
result = surus.extract_to_json(
    document, 
    custom_prompt="Extract only financial data as JSON...",
    temperature=0.1
)
```
Task optimization without model knowledge

**Layer 4: Full Control**
```python
result = surus.extract_to_json(
    document,
    model_override="claude-3-opus",
    prompt_module=None,  # Remove default prompting
    custom_prompt="<your_complete_prompt>",
    temperature=0.1,
    max_tokens=4000
)
```
Expert control

**vs Alternatives:**
- AWS SDK: Starts simple, grows complex when needed
- Rails: No lock-in, experts get full power

## Composable Prompts

**Fixed Prompts Fail:**
- Domain differences
- Specific requirements
- Evolution needs

**Full Custom Problems:**
- Requires expertise
- Reinventing wheel
- No community benefit

**SURUS Approach:**

```python
# Append to base prompt
surus.summarize(
    text,
    prompt_append="Focus on financial implications and risks."
)

# Replace base prompt entirely
surus.summarize(
    text,
    prompt_module="financial_summary",  # Community/custom module
    custom_prompt="Your expert financial analysis prompt..."
)
```

**Benefits:**
- Community contributions
- Domain specialization
- Continuous improvement
- Backward compatibility

## Engine Architecture

**Separate Engines:**
- TextEngine: text tasks
- AudioEngine: audio tasks
- VisionEngine: vision tasks

**Benefits:**
- Domain optimization
- Independent evolution
- Resource specialization
- Team ownership
- Clear boundaries

## Evolution Phases

**Phase 1: MVP**
- Basic task verbs work Day 1
- Success: Users solve real problems immediately

**Phase 2: Performance**
- Add high_performance flag
- Success: Users pay more for quality

**Phase 3: Customization**
- Custom prompts
- Success: Users stop switching to direct APIs

**Phase 4: Expert Control**
- Model overrides
- Success: Default choice for all user types

**Phase 5: Ecosystem**
- Community modules
- Industry-specific verbs
- Plugin architecture

## vs Competitors

**vs OpenAI SDK:** Task abstraction reduces cognitive load
**vs LangChain:** Simpler mental model, faster productivity
**vs Replicate/Together:** Curated vs choice overload
**vs Direct APIs:** Higher abstraction, future-proofing

## Success Metrics

**Week 1:** Real tasks solved immediately, zero docs needed
**Month 1:** Users choose high_performance, request new verbs
**Quarter 1:** Community contributions, power user adoption
**Year 1:** Industry standard, ecosystem growth

**Core Success:** Time to value + infinite ceiling
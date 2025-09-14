# CLAUDE.md

----
<main_rule>
**MAIN RULE FOR CLAUDE**: 
Every writing should have the high signal/noise ratio and minimalism at a guiding principle. 
Less is More
Information density is the main metric to optimize
Bluff is to be minimized at all costs. 
</main_rule>
---


This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SURUS AI is a Python library designed to provide a unified interface for AI models across audio, text, vision, and embeddings - similar to Together AI, OpenAI, Anthropic, and Replicate.

## Core Philosophy

SURUS follows a "Task-oriented" AI approach where AI models are viewed as **learned programs** that perform **tasks**. The library focuses on:

1. **Task-first UX**: Users want to solve tasks (transcribe, summarize, extract_to_json, chat, annotate) without caring about the underlying model
2. **Two-tier performance**: Each task verb has a `high_performance=False` parameter that can be set to `True` for better performance at higher cost (switches to more powerful model)
3. **Progressive complexity disclosure**: Start simple, allow more control through low-level parameters when needed
4. **Composable prompt modules**: Users can append to base prompt modules or replace them entirely

## Architecture Principles

- **Pragmatic and composable**: Develop basics quickly, progressively add support for new verbs
- **Two models per task maximum**: Good model (default) and best model (high_performance=True)
- **Modular prompt system**: Base prompt modules can be composed, extended, or replaced
- **Unified interface**: Abstract away model-specific implementations behind task verbs

## Example Usage Pattern

```python
surus.transcribe()  # Uses default transcription model + prompt module
surus.transcribe(high_performance=True)  # Uses best model for higher accuracy
```

## Development Approach

The codebase is designed to be built iteratively:
1. Implement basic functionality quickly (target: 1 day for basics)
2. Progressively add new task verbs
3. Add support for different backend models per task
4. Develop and refine prompt modules for each task
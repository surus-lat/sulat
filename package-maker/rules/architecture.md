# Architecture Rules

## Design Principles
- Task-first: Users think in problems, not models
- Two-tier performance: Good/Best, never more
- Progressive disclosure: Simple → Powerful, never the reverse
- Composable: Build complexity through combination, not configuration

## Decision Framework
- Does this reduce cognitive load?
- Can beginners succeed immediately?
- Can experts get full control eventually?
- Does this scale with user sophistication?

## Anti-Patterns
- Model-first interfaces
- Configuration explosion
- Premature abstraction
- Feature parity with underlying APIs

## Validation Tests
- Can user solve real problem in 1 line?
- Does high_performance flag provide clear value?
- Can experts override everything?
- Does architecture support evolution without breaking changes?
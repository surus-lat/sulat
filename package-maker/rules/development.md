# Development Rules

## Implementation Priority
1. Basic functionality that works Day 1
2. High-performance tier
3. Customization layer
4. Expert controls

## Code Quality
- Every function has single clear purpose
- No abstraction without 3+ concrete use cases
- Optimize for reading, not writing
- Fail fast with clear error messages

## Testing Strategy
- Test user workflows, not implementation details
- Each task verb must have integration test
- Performance benchmarks for both tiers
- Backward compatibility validation

## Evolution Rules
- New features cannot break existing simple usage
- Complex features live in separate layers
- Default behavior never changes
- Deprecation requires migration path
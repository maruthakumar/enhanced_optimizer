# BMad Method Command Reference Sheet

## Installation Commands

### Project Setup
```bash
# Interactive installation (recommended)
npx bmad-method install

# Flatten project into single file for web agents
npx bmad-method flatten
```

## Web UI Commands

### Planning Phase Commands
```bash
# Help command
/help

# Project Management Commands
@pm
*create-brownfield-prd         # Create PRD for existing project enhancement
*create-brownfield-epic        # Create single epic without full PRD
*create-brownfield-story       # Create single story for small changes

# Architecture Commands  
@architect
*document-project              # Document existing codebase
*create-brownfield-architecture # Create architecture with integration focus

# UX Commands (if needed)
@ux
# Create front-end specifications
# Generate UI prompts for Lovable/V0

# Product Owner Commands
@po
*execute-checklist-po          # Run master checklist for document alignment
*shard-doc                     # Shard PRD/Architecture documents
```

## IDE Integration Commands

### Agent Interaction in IDE

#### For IDEs with @ Symbol (Cursor, Windsurf)
```bash
@pm Create a PRD for a task management app
@architect Design the system architecture  
@dev Implement the user authentication
@qa Review the implementation
@sm Draft next story from sharded epic
```

#### For IDEs with Slash Commands (Claude Code)
```bash
/pm Create user stories
/dev Fix the login bug
/architect Update system design
/qa Review code quality
```

### Development Cycle Commands

#### Scrum Master Commands
```bash
@sm
# Reviews previous story dev/QA notes
# Drafts next story from sharded epic + architecture
```

#### Developer Commands
```bash
@dev
# Sequential task execution
# Implement tasks + tests
# Run all validations
# Mark ready for review + add notes
```

#### QA Commands
```bash
@qa
# Review story draft (optional)
# Review story against artifacts
# Senior dev review + active refactoring
# Review, refactor code, add tests, document notes
```

## Document Sharding Commands

```bash
# In IDE after saving documents
@po
shard docs/brownfield-prd.md
shard docs/brownfield-architecture.md
```

## Special Agent Commands

### BMad-Master Agent
Can execute any command from any agent except actual story implementation.

### BMad-Orchestrator Agent  
Web-only agent for team facilitation. Do not use in IDE.

## Brownfield Workflow Quick Reference

### Approach A: PRD-First (Recommended for Large Projects)
1. `@pm → *create-brownfield-prd`
2. `@architect → *document-project` (focused on PRD areas)
3. `@architect → *create-brownfield-architecture`
4. `@po → *execute-checklist-po`
5. Save documents and shard in IDE

### Approach B: Document-First (For Smaller Projects)  
1. `@architect → *document-project` (entire codebase)
2. `@pm → *create-brownfield-prd`
3. `@architect → *create-brownfield-architecture`
4. `@po → *execute-checklist-po`
5. Save documents and shard in IDE

### Quick Enhancement Options
- Major changes: Use full brownfield workflow
- Single epic: `@pm → *create-brownfield-epic`
- Bug fix/tiny feature: `@pm → *create-brownfield-story`

## Development Modes
- **Incremental Mode**: Step-by-step with user input
- **YOLO Mode**: Rapid generation with minimal interaction

## Important Notes

### Critical Transition Points
- After planning in web UI, switch to IDE for development
- Always commit changes before proceeding to next story
- Run regression tests and linting before marking story complete

### File Locations
- PRD: `docs/prd.md` or `docs/brownfield-prd.md`
- Architecture: `docs/architecture.md` or `docs/brownfield-architecture.md`
- Technical preferences: `.bmad-core/data/technical-preferences.md`
- Core config: `bmad-core/core-config.yaml`

### Best Practices
- Keep context lean and focused
- Use appropriate agent for each task
- Work in small, focused iterations
- Commit regularly
- Follow existing code patterns and conventions
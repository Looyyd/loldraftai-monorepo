# DraftKing Development Guidelines

## Build/Lint/Test Commands
- Root: `yarn build`, `yarn dev`, `yarn lint` (runs for all workspaces)
- Web: `yarn dev` (Next.js), `yarn build`, `yarn start`, `yarn lint`
- Desktop: `yarn dev` (Vite), `yarn build`, `yarn build:dir`, `yarn lint`
- Data: `yarn dev` (tsx), `yarn lint`
- ML: `python scripts/match-prediction/prepare_data.py`, `python serve-model.py`

## Code Style Guidelines
- **TypeScript**: Strict typing, ES2022 target, ESNext modules
- **Imports**: ESM modules preferred, ordered imports
- **Formatting**: Prettier, consistent spacing/braces
- **Naming**: camelCase for variables/functions, PascalCase for classes/components
- **Components**: Functional components with hooks, typed props interfaces
- **Error Handling**: Try/catch blocks with proper logging, typed errors
- **State Management**: React hooks, context where appropriate
- **Python**: Type hints, docstrings, exception handling with status codes

## Project Structure
- Monorepo with workspaces in `apps/*` and `packages/*`
- Shared packages: UI components, ESLint/TS configs, API clients, database utilities
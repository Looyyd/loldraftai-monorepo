# LoLDraftAI Desktop Application

## Overview

LoLDraftAI's desktop application is an Electron-based client that provides League of Legends draft analysis with live game integration. Built with Vite, React, and TypeScript, it shares core components with the web frontend(@draftking/web-frontend) through a shared UI (@draftking/ui) package.
It is automatically build with a github action, see `./github/workflows/desktop-build.yml`

## Technical Stack

### Core Technologies

- **Framework**: Electron + Vite + React
- **Language**: TypeScript
- **UI Components**: Shared @draftking/ui package
- **State Management**: Zustand
- **Storage**: electron-store
- **Auto Updates**: electron-updater

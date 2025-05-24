# LoLDraftAI Web Frontend

Code for the website https://loldraftai.com/

## Overview

LoLDraftAI's web frontend is a Next.js application. It uses the machine learning model trained in `./apps/machine-learning` to provide real-time draft predictions and champion recommendations.
it shares core components with the desktop application(@draftking/desktop) through a shared UI (@draftking/ui) package.

It is hosted on Vercel with heavy media hosted on Cloudflare R2.

### Technical Stack

#### Core Technologies

- **Framework**: Next.js 15.0
- **Language**: TypeScript
- **Styling**: Tailwind CSS with custom theme system
- **UI Components**: Custom component library (@draftking/ui), which heavily use shadcn.

### Setup

```bash
# Install dependencies
yarn install

# Start docker redis database(for rate limits)
yarn docker:up

# Run development server
yarn dev


```

## Upload images to cloudflare

A one time script used to upload heavy media to cloudflare.

```bash
cd public
find . -type f \( -name "*.webp" -o -name "*.png" -o -name "*.svg" -o -name "*.ico" \) -exec yarn wrangler r2 object put loldraftai-web-media-files/{} --file "$(pwd)/{}" \;
```

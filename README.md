# LoLDraftAI monorepo

full source code for the website https://loldraftai.com/

## High level repo overview:

./apps/data-collection:
Data collection scripts, run on a VM and store data in a postgresql database.
./apps/desktop:
Code for the desktop application
./apps/machine-learning:
Code for the machine learning part contains code for:

- data download and preperation
- main model train
- pro model train
- model hosting

./apps/team-comp-visualizer:
Code for a proof of concept desktop app that uses millions of team comps rated by the pro finetuned model, to see which are the best.
See https://docs.google.com/document/d/1aHmNZq_Wvn6YChEOKfZa1i1fs_97NExXwBnLYN1WInI/edit?usp=sharing for how it looks like.

./apps/web-frontend:
Code for the LoLDraftAI website.

./packages/ui:
UI package, used by both the desktop and web-frontend app.

# LoLDraftAI monorepo

Full source code for the website https://loldraftai.com/

## Repository Structure

### Apps

- **data-collection**  
  Data collection scripts, run on a VM and store data in a postgresql database.

- **desktop**  
  Code for the desktop application. Can be downloaded from https://loldraftai.com/download

- **machine-learning**  
  Code for the machine learning part. See `./apps/machine-learning/README.md` for full details.

- **team-comp-visualizer**  
  Code for a proof of concept desktop app that uses millions of team comps rated by the pro finetuned model, to see which are the best.  
  See https://docs.google.com/document/d/1aHmNZq_Wvn6YChEOKfZa1i1fs_97NExXwBnLYN1WInI/edit?usp=sharing for how it looks like.

- **web-frontend**  
  Code for the LoLDraftAI website.

### Packages

- **ui**  
  UI package, used by both the desktop and web-frontend app.

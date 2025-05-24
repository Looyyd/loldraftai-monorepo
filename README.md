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

## Infrastructure

The infrastructure used for this project is:

- An Azure VM(Standard D2s v3) for running the data-collection scripts.

- A Postgresql database(Standard_B4ms (4 vCores)) for storing the league api data(Summoners and Matches). Complete matches are then exported to an Azure bucket in parquet format.

- Cloudflare for large media hosting(download file for desktop and images for web-frontend).

- Vercel for web-frontend.

- Azure container apps for the model inference(0.5vCPU and 1Gb Ram was enough).

- My gaming PC for model training :-)

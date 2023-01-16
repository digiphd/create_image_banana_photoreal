
# 🍌 Modified Banana Serverless GPU processing

This repo gives a basic framework for serving Stable Diffusion [Photoreal Model](https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0) model in production using simple HTTP servers. It is capable of returning up to 3 images at 512x512 without going over the 16GB hard limit on Banana GPU's

Please note that this model can create Adult content. So be careful what you type. 

# Quickstart

If you want to customize beyond the prebuilt model:

**[Follow the quickstart guide in Banana's documentation to use this repo](https://docs.banana.dev/banana-docs/quickstart).** 

*(choose "GitHub Repository" deployment method)*

### Additional Steps (outside of quickstart guide)

1. Create your own private repo and copy the files from this template repo into it. You'll want a private repo so that your huggingface keys are secure or place it as an ENV variable called "ENV HF_AUTH_TOKEN".
2. Edit the `dockerfile` in your forked repo with `ENV HF_AUTH_TOKEN=your_auth_token`


# Helpful Links
Understand the 🍌 [Serverless framework](https://docs.banana.dev/banana-docs/core-concepts/inference-server/serverless-framework) and functionality of each file within it.

Generalize this framework to [deploy anything on Banana](https://docs.banana.dev/banana-docs/resources/how-to-serve-anything-on-banana).


<br>



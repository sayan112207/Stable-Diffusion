# Stable-Diffusion

## Overview

### Problem Statement: Deep Learning-based Image Generation with Text Prompts

Generate images from text prompts and assemble them into a grid. In each 2×2 block, one image is uniquely generated from a user-provided prompt while three filler images come from random artistic prompts. The project evaluates semantic alignment using CLIP and demonstrates trade-offs between semantic fidelity and stylistic adaptation.

This repository implements deep learning–based image generation from text prompts using Stable Diffusion. Three distinct approaches are demonstrated:

- **Approach 1:** Using a Pre-Trained Stable Diffusion model with filler images sourced from the Flickr8k dataset.
- **Approach 2:** Using a Fine-Tuned Diffusion model (stabilityai/sd-turbo) fine-tuned with LoRA on Impressionism-style images (from WikiArt).
- **Approach 3:** Generating all images using Stable Diffusion—where the unique image is generated from the main prompt, and filler images are generated via random artistic prompts created by a text-generation API.


### Methodology

- **Approach 1: Pre-Trained Model**  
  - **Model:** `CompVis/stable-diffusion-v1-4`  
  - **Filler Images:** Sourced from the Flickr8k dataset  
  - **Result:** Baseline outputs with an average CLIP score around ~0.32.<br>

- **Approach 2: Fine-Tuned Model with LoRA**  
  - **Model:** `stabilityai/sd-turbo` fine-tuned with LoRA (r=8, lora_alpha=16, lora_dropout=0.1) on [Impressionism images from WikiArt](https://huggingface.co/datasets/huggan/wikiart)  
  - **Result:** More stylistically consistent (pencil-sketch impressionism) images with an average CLIP score around ~0.31, which increases to ~0.35 when using defined prompts with specific design elements.
  - **Training:** Early stopping is implemented based on epoch loss trends.<br>

- **Approach 3: Fully AI-Generated Filler Images**  
  - **Model:** `CompVis/stable-diffusion-v1-4` used for all image generations  
  - **Filler Prompts:** Random artistic prompts are generated dynamically using a text-generation API (via Gemini API)  
  - **Result:** A grid of 64 images where each 2×2 block contains one unique image from the main prompt and three filler images generated on-the-fly from diverse artistic prompt categories (e.g., animal, landscape, object, cityscape, nature, space, abstract).<br>

---

## Usage and Running Instructions
Two main notebooks are provided:

- [DeepAlgoAssignment.ipynb](https://github.com/sayan112207/Stable-Diffusion/blob/main/DeepAlgoAssignment.ipynb): Contains the complete pipeline for image generation, grid assembly, and CLIP evaluation.
  - Approach 1: Pre-Trained Model:
    - No installation is required just run the cells, the flickr8k dataset will be downloaded automatically.<br>
  - Approach 2: Fine-Tuned Model with LoRA:
    - Create a folder named `LORA Model` in the colab workspace and inside the folder upload the files in [LORA Model](https://github.com/sayan112207/Stable-Diffusion/tree/main/LORA%20Model) and run the cells.<br>
  - Approach 3: Fully AI-Generated Filler Images:
    - Setup your [Gemini API Key](https://aistudio.google.com/app/apikey) named as `GOOGLE_API_KEY` in Colab Secrets and you're good to go.<br>
- [Fine_Tuning_Stable_Diffusion.ipynb](https://github.com/sayan112207/Stable-Diffusion/blob/main/Fine_Tuning_Stable_Diffusion.ipynb): Focuses on fine-tuning the model using LoRA on the Impressionism subset of WikiArt.

---

## **Fine-Tuning Training Overview with LORA:**

- **Base Model:** `stabilityai/sd-turbo`  
- **Fine-Tuning Dataset:** Filtered Impressionism images from the WikiArt collection  
- **Fine-Tuning Technique:** LoRA is applied to specific modules (`to_q`, `to_k`, `to_v`, `to_out.0`) with configuration:  
  - **r:** 8  
  - **lora_alpha:** 16  
  - **lora_dropout:** 0.1  
- **Training Details:**
  - **Epochs:** 30 (training was halted early based on early stopping criteria)
  - **Batch Size:** 1
  - **Learning Rate:** 1e-4
  - **Early Stopping:** Implemented if the average loss does not improve for 2 consecutive epochs.  
  - **Training Loss Trend:** Loss decreases until Epoch 3 (~0.19) and then increases, triggering early stopping.  
- **Outputs:**
  - Fine-tuned model checkpoints saved in the `LORA Model` directory.
  - Training loss plot (`training_loss.png`) showing the epoch-wise loss trend.
    ![training loss plot](https://github.com/sayan112207/Stable-Diffusion/blob/main/training_loss.png?raw=true)

---

## Results

### Approach 1 (Pre-Trained Model)

- **Main Prompt:** `a cat wearing neon-glasses`
- **Average CLIP Score:** ~0.32
- **Visual Output:** A 4×4 grid where each block contains one unique image (generated from the prompt) and three filler images (sourced from the Flickr8k dataset).
  ![o/p grid of approach 1](https://github.com/sayan112207/Stable-Diffusion/blob/main/output%20grid%20with%20pretrained_model.png?raw=true)  

### Approach 2 (Fine-Tuned Model with LoRA)

- **Main Prompt:** `a cat wearing neon-glasses`
- **Average CLIP Score:** ~0.31 (increases to ~0.35 when defined prompts with specific design elements are used)  
- **Training Loss:**  
  - Loss decreases until Epoch 3 (~0.19) and then increases, confirming the effectiveness of early stopping to prevent overfitting.  
- **Visual Output:** A grid with images exhibiting a pencil-sketch Impressionist style.
  ![o/p grid of approach 2](https://github.com/sayan112207/Stable-Diffusion/blob/main/output%20grid%20with%20finetuned_model.png?raw=true)

### Approach 3 (Fully AI-Generated Filler Images)

- **Main Prompt:** `a cat wearing neon-glasses`   
- **Filler Prompts:** Dynamically generated from diverse categories (animal, landscape, object, cityscape, nature, space, abstract) using a text-generation API. These prompts are printed to the console during generation.
- **Visual Output:** A 4×4 grid (64 images total) where each 2×2 block contains one unique image (from the main prompt) and three filler images generated on-the-fly via random artistic prompts.
  ![o/p grid of approach 3](https://github.com/sayan112207/Stable-Diffusion/blob/main/output%20grid%20with%20all%20AI%20images.png?raw=true)

---

## References

- **Stable Diffusion:**  [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- **Fine-Tuned Model:**  [stabilityai/sd-turbo](https://huggingface.co/stabilityai/sd-turbo)
- **WikiArt Dataset (Impressionism):**  [huggan/wikiart](https://huggingface.co/datasets/huggan/wikiart/viewer/default/train?f[style][value]=12)
- **Flickr8k Dataset:** Used for filler images in Approach 1 and 2.
- **CLIP:**  [OpenAI CLIP](https://github.com/openai/CLIP)
- **Gemini API:** Used for generating random artistic prompts in Approach 3.

# 🎭 Multimodal Story & Poem Generator

A **Multimodal Generative AI application** that generates creative **stories or poems** from an uploaded image.  
The system combines **image captioning, text generation, and speech synthesis** to produce both text and audio outputs.

---

## 📌 Project Description

This project allows users to:
- Upload an image
- Enter a theme (example: Love, Mystery, Friendship, Fantasy)
- Add optional characters
- Choose content type (**Story** or **Poem**)
- Generate:
  - AI-generated text
  - Downloadable `.txt` file
  - Audio narration (`.mp3`)

The app is built using **Gradio**, powered by **BLIP** for image captioning and **Falcon RW-1B** for text generation.

---

## 🚀 Features

- 🖼️ Image upload support  
- 🧠 Automatic image caption generation using BLIP  
- ✍️ Story/Poem generation using a language model  
- 🎨 Theme-based creative generation  
- 🧑‍🤝‍🧑 Optional character customization  
- 📄 Download generated output as `.txt` file  
- 🔊 Audio narration using gTTS  
- 🌍 Multi-language audio support  

---

## 🌍 Supported Languages (Audio)

- English
- Hindi
- Tamil
- Telugu
- Malayalam
- Kannada
- Marathi
- Bengali

---

## 🧠 Tech Stack

- Python  
- Gradio  
- HuggingFace Transformers  
- PyTorch  
- BLIP Image Captioning Model  
- Falcon RW-1B Text Generation Model  
- gTTS (Google Text-to-Speech)  
- Pillow  

---

## ⚙️ Project Workflow

1. User uploads an image
2. BLIP model generates an image caption
3. Caption + Theme + Characters are combined into a prompt
4. Falcon RW-1B generates a Story/Poem
5. Text is saved into a `.txt` file
6. gTTS converts the generated text into an `.mp3` file
7. Outputs are displayed and available for download

---

## 📂 Project Structure

├── app.py
├── requirements.txt
└── README.md


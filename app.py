import gradio as gr
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from gtts import gTTS
import tempfile
import os

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP for image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Load language model (falcon-rw-1b preferred, fallback to gpt2)
gpt_tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b", trust_remote_code=True)
gpt_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b", trust_remote_code=True).to(device)

# Map for gTTS language codes
LANG_CODE_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Malayalam": "ml",
    "Kannada": "kn",
    "Marathi": "mr",
    "Bengali": "bn"
}

# Generate caption from image
def generate_caption(image):
    inputs = blip_processor(image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

# Generate story or poem
def generate_text(caption, theme, characters, language, content_type):
    if content_type.lower() == "story":
        prompt = f"{caption}. This inspired a story about {theme.lower()}"
        if characters:
            prompt += f" involving {characters}"
        prompt += ". It begins like this:\n"
    else:  # poem
        prompt = f"{caption}. A poem themed around '{theme}'"
        if characters:
            prompt += f", mentioning {characters}"
        prompt += ":\n"

    input_ids = gpt_tokenizer.encode(prompt, return_tensors="pt").to(device)

    output_ids = gpt_model.generate(
        input_ids,
        max_length=250,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        eos_token_id=gpt_tokenizer.eos_token_id,
        pad_token_id=gpt_tokenizer.pad_token_id or gpt_tokenizer.eos_token_id
    )

    output = gpt_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output[len(prompt):].strip()

# Main function
def generate_output(image, theme, characters, language, content_type):
    if language not in LANG_CODE_MAP:
        return "Unsupported language", None, None

    caption = generate_caption(image)
    generated_text = generate_text(caption, theme, characters, language, content_type)

    # Save text to file
    txt_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8")
    txt_file.write(generated_text)
    txt_file.close()

    # Generate audio with gTTS
    lang_code = LANG_CODE_MAP[language]
    tts = gTTS(text=generated_text, lang=lang_code)
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(audio_path)

    return generated_text, txt_file.name, audio_path

# UI with Gradio Blocks
with gr.Blocks(title="Multimodal Story & Poem Generator") as demo:
    gr.Markdown("## 🎭 Multimodal Story & Poem Generator")
    gr.Markdown("Upload an image, choose a theme and language, and get a creative story or poem with audio!")

    with gr.Row():
        image = gr.Image(type="pil", label="🖼️ Upload Image")
       

    with gr.Row():
        theme = gr.Textbox(label="🎨 Enter a Theme (e.g., Friendship, Mystery, Dreams)")
        characters = gr.Textbox(label="🧑‍🤝‍🧑 Characters (Optional)")

    content_type = gr.Radio(["Poem", "Story"], label="📝 Choose Content Type")
    generate_btn = gr.Button("✨ Generate")

    output_text = gr.Textbox(label="📜 Generated Text", lines=10)
    txt_file = gr.File(label="📄 Download .txt")
    audio_file = gr.Audio(label="🔊 Listen / Download Audio")

    generate_btn.click(
        fn=generate_output,
        inputs=[image, theme, characters, language, content_type],
        outputs=[output_text, txt_file, audio_file]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()



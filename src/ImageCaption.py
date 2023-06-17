import gradio as gr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import List
from dataclasses import dataclass
from args import args 

class ImageCaption:
  def __init__(self):
    self.processor = BlipProcessor.from_pretrained(args.model_path)
    self.model = BlipForConditionalGeneration.from_pretrained(args.model_path)
    self.caption_generator = self.caption_generator


  def caption_generator(self, image, num_captions) -> List[str]:
      num_captions = int(float(num_captions))
      raw_image = Image.fromarray(image).convert('RGB')
      inputs = self.processor(raw_image, return_tensors="pt")
      out = self.model.generate(
          **inputs,
          num_return_sequences=num_captions,
          max_length=32,
          early_stopping=True,
          num_beams=num_captions,
          no_repeat_ngram_size=2,
          length_penalty=0.8
      )
      captions = []
      for i, caption in enumerate(out):
          captions.append(self.processor.decode(caption, skip_special_tokens=True))
      return captions 

  def photo_upload(self, photo, num_captions) -> List[str]:
      captions = self.caption_generator(photo, num_captions)
      return captions

# # Define the inputs and outputs for Gradio interface
# photo_input = gr.inputs.Image(label="Upload Photo")
# num_captions_input = gr.inputs.Dropdown([1, 2, 3, 4, 5], label="Select number of captions to generate")
# caption_output = gr.outputs.Textbox(label="Captions")

# # Create the Gradio interface
# ImgCap = ImageCaption()
# # interface = gr.Interface(fn=photo_upload, inputs=[photo_input, num_captions_input], outputs=caption_output, 
# #                         title="AI Image Captioning", sidebar=True)

# # # Launch the interface
# # interface.launch(share = True)

# if __name__ == '__main__':
#   ImgCap = ImageCaption()
#   print(ImgCap.photo_upload(photo, num_captions))
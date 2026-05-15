# pip install gradio==4.44.1
import gradio as gr
from inference import Inference

# modalities seem have little effect on the results 
TEXT_OPTIONS = ["CFP", "UWF", "FFA", "SLO", "OCTA"]

inference_engine = Inference(model_path='checkpoints/UNet_DCP_1024')

def main(image, text):
    out = inference_engine.inference(image, text)
    return out

interface = gr.Interface(
    fn=main,
    inputs=[
        gr.Image(type="numpy"),
        gr.Dropdown(
            choices=TEXT_OPTIONS,
            label="Modality",
            value=TEXT_OPTIONS[0]
            )
        ],
    outputs=gr.Image(type="numpy"),
    title="Broad domain retinal vessel segmentation",
    description=""
)
interface.launch(server_name='127.0.0.1', server_port=8080)

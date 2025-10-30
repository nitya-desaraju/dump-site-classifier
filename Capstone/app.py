import gradio as gr
import numpy as np
from PIL import Image
# from pipeline import model 

def fake_model(image):
    if image is None:
        return None, "Error", 0.0

    heatmap = image.copy()
    overlay = Image.new('RGBA', image.size, (255, 0, 0, 95))
    heatmap = Image.alpha_composite(image.convert('RGBA'), overlay)

    class_name = "Plastic"
    confidence = 0.92

    return heatmap, class_name, confidence

def display(image):
    if image is None:
        return None, None, "Please upload an image."

    heatmap, class_name, confidence = fake_model(image)
    category = {class_name: confidence}

    return image, heatmap, category


pickleball_theme = gr.themes.Base(
    primary_hue="green",
    secondary_hue="cyan",
    neutral_hue="slate",
)

# Custom CSS for all styling
custom_css = """
<style>
/* Translucent Blue Background */
body {
    background-color: rgba(5, 201, 240, 0.1) !important;
}

/* Green border around the main component boxes */
#input_image_box, #class_output_box, #heatmap_output_box {
    border: 2px solid #22c55d !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
}

/* Green color for the footer text */
#footer_text strong {
    color: #22c55d !important;
}

/* Custom color and border for the primary button */
.gradio-container .gr-button-primary {
    background-color: #22c55d !important;
    border: 2px solid #475569 !important; /* Dark slate border to match "Results" text */
}

/* Vertical divider between the two main columns */
#results_column {
    border-left: 1px solid #cbd5e1; /* Light slate color for the divider */
    padding-left: 20px;
}
</style>
"""

with gr.Blocks(theme=pickleball_theme) as demo:
    # Inject the custom CSS into the app
    gr.HTML(custom_css)
    
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1 style="font-size: 3em; margin-bottom: 0;">♻️ Pickle V 🥒</h1>
            <p style="margin-top: 0.5em; font-size: 1.2em;">Keeping our pickleball courts and planet clean!</p>
        </div>
        """
    )

    # Added the description block
    gr.Markdown(
        """
        <div style="text-align: center; max-width: 800px; margin: auto; padding-bottom: 20px;">
        <p><strong>Welcome to Pickle V!</strong> This machine learning model receives a satellite image of an illegal dump site and identifies where the waste is and what kind of waste it is. This can help city management find the best route to remove it and keep the city clean for the people and environment!</p>
        </div>
        """
    )


    with gr.Row():
        with gr.Column(scale=1):
            # Added elem_id for CSS targeting
            input_image = gr.Image(type="pil", label="Upload Satellite Image", sources=["upload"], height=400, width=400, elem_id="input_image_box")
            submit_button = gr.Button("Predict", variant="primary")

        # Added elem_id to the column for the vertical divider
        with gr.Column(scale=2, elem_id="results_column"):
            gr.Markdown("## Results")
            # Added elem_id for CSS targeting
            class_output = gr.Label(label="Waste Classification", elem_id="class_output_box")
            gr.Markdown("---")
            # Added elem_id for CSS targeting
            heatmap_output = gr.Image(label="Dump Site Heatmap", height=300, width=300, elem_id="heatmap_output_box")

    submit_button.click(
        fn=display,
        inputs=input_image,
        outputs=[input_image, heatmap_output, class_output]
    )

    gr.Examples(
        label="Click on an example to try it out!",
        examples=[
            ["dump_site_1.jpg"],
            ["dump_site_2.jpg"],
        ],
        inputs=input_image,
        outputs=[input_image, heatmap_output, class_output],
        fn=display,
        cache_examples=False,
    )

    gr.Markdown("---")

    gr.Markdown(
        """
        <p id="footer_text" style='text-align: center; font-style: italic;'>
        Brought to you by the <strong>Pickleball Lovers</strong>
        </p>
        """
    )


if __name__ == "__main__":
    try:
        import os
        if not os.path.exists("dump_site_1.jpg"):
            Image.new('RGB', (400, 400), color = 'purple').save("dump_site_1.jpg")
        if not os.path.exists("dump_site_2.jpg"):
            Image.new('RGB', (400, 400), color = 'green').save("dump_site_2.jpg")
    except Exception as e:
        print(f"Could not create example images: {e}")

    demo.launch()
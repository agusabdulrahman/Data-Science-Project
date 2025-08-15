# import the necessary libraries
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import os
from dotenv import load_dotenv

load_dotenv()

# Inisialisasi client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

# define a prediction function to infer on an image
def infer_image(image, confidence, iou_threshold):
    # save the uploaded or captured image to a file
    image_path = "uploaded_image.jpg"
    image.save(image_path)
    
    # set custom configuration
    custom_configuration = InferenceConfiguration(confidence_threshold=confidence, iou_threshold=iou_threshold)

    # infer on the image using the client
    with CLIENT.use_configuration(custom_configuration):
        result = CLIENT.infer(image_path, model_id="construction-safety-gsnvb/1")
    
    # extract predictions
    predictions = result.get('predictions', [])
    
    # define a color map for different classes of the model
    class_colors = {
        "helmet": "red",
        "person": "blue",
        "vest": "yellow",
        "no-helmet": "purple",
        "no-vest": "orange"
        # Add other classes and their corresponding colors here
        # "class_name": "color",
    }
    
    # draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
        
    for pred in predictions:
        x = pred['x']
        y = pred['y']
        width = pred['width']
        height = pred['height']
        left = x - width / 2
        top = y - height / 2
        right = x + width / 2
        bottom = y + height / 2
        
        # get the color for the class
        color = class_colors.get(pred['class'], "green")  # default to green if class is not in the color map
        
        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        
        # Draw the label
        label = f"{pred['class']} ({pred['confidence']:.2f})"
        text_size = draw.textbbox((0, 0), label, font=font)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]
        text_background = [(left, top - text_height - 4), (left + text_width + 4, top)]
        draw.rectangle(text_background, fill=color)
        draw.text((left + 2, top - text_height - 2), label, fill="white", font=font)
    
    return image, str(predictions)

# create a Gradio interface
myInterface = gr.Interface(
    fn=infer_image,  # function to process the input
    inputs=[
        gr.Image(type="pil"),  # input type is an image
        gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Confidence Threshold"),
        gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="IoU Threshold")
    ],  
    outputs=[
        gr.Image(type="pil"),  # output is an image
        gr.Textbox(label="Predictions")  # output is text
    ],
    title="Construction Safety Detection",
    description="Upload an image to detect. Adjust the confidence and IoU thresholds.",
)

# launch the Gradio app
if __name__ == "__main__":
    myInterface.launch()

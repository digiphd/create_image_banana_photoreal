import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, AutoencoderKL
import base64
from io import BytesIO
import os


NEGATIVE_PROMPT="floating limbs, disconnected limbs, kitsch, cartoon, fake, boring, long neck, out of frame, extra fingers, mutated hands, monochrome, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, glitchy, bokeh, (((long neck))), (child), (childlike), ((flat chested)), red eyes, multiple subjects, extra heads, close up, man, asian, text, bad anatomy, morphing, messy broken legs decay, ((simple background)), deformed body, lowres, bad anatomy, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low jpeg artifacts, signature, watermark, username, blurry, out of focus, old, amateur drawing, odd, fat, morphing, black and white, ((simple background)), artifacts, signature, artist name, [blurry], disfigured, mutated, (poorly hands), messy broken legs, decay, painting, duplicate, closeup",

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")
    
    # this will substitute the default PNDM scheduler for K-LMS  
    lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

    # model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=lms, use_auth_token=HF_AUTH_TOKEN).to("cuda")
    # model = StableDiffusionPipeline.from_pretrained("hassanblend/HassanBlend1.5", safety_checker=None, scheduler=lms, use_auth_token=HF_AUTH_TOKEN).to("cuda")
    model = StableDiffusionPipeline.from_pretrained("dreamlike-art/dreamlike-photoreal-2.0", safety_checker=None, scheduler=lms, use_auth_token=HF_AUTH_TOKEN).to("cuda")

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    "test"
    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    negative_prompt = model_inputs.get('negative_prompt', None)
    height = model_inputs.get('height', 512)
    width = model_inputs.get('width', 512)
    num_inference_steps = model_inputs.get('num_inference_steps', 50)
    guidance_scale = model_inputs.get('guidance_scale', 7.5)
    input_seed = model_inputs.get("seed",None)

    #If "seed" is not sent, we won't specify a seed in the call
    generator = None
    if input_seed != None:
        generator = torch.Generator("cuda").manual_seed(input_seed)

    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    with autocast("cuda"):
        images = model(prompt, negative_prompt=negative_prompt, height=height,width=width,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,generator=generator).images[0]
    print("number of images:",len(images))

    buffered = BytesIO()
    images.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    images_base64 = []
    for image in images:

        buffered = BytesIO()
        image.save(buffered,format="JPEG")
        images_base64.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))
    print(len(images_base64))
    # Return the results as a dictionary
    return {'images_base64': images_base64, 'image_base64': image_base64}

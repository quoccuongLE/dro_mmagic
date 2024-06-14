from mmagic.apis import MMagicInferencer

# Create a MMEdit instance and infer
editor = MMagicInferencer(model_name="controlnet_animation")

prompt = "a girl and a boy, black hair, best quality, extremely detailed"
negative_prompt = (
    "longbody, lowres, bad anatomy, bad hands, missing fingers, "
    + "extra digit, fewer digits, cropped, worst quality, low quality"
)

# you can download the example video with this link
# https://user-images.githubusercontent.com/12782558/227418400-80ad9123-7f8e-4c1a-8e19-0892ebad2a4f.mp4
# video = "/path/to/your/input/video.mp4"
# save_path = "/path/to/your/output/video.mp4"
image = "./data/IMG_1294.jpg"
save_path = "output/anime/"

# Do the inference to get result
editor.infer(
    img=image, prompt=prompt, negative_prompt=negative_prompt, save_path=save_path,
    strength=0.5,
    num_inference_steps=40,
    controlnet_conditioning_scale=0.5,
    image_width=1024,
    image_height=1024
    
)

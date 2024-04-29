from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
)  # doctest: +IGNORE_RESULT

#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#image = Image.open(requests.get(url, stream=True).raw)
#print(type(image))
# Q300001.mp4

from decord import VideoReader
video_fname = "orange.mp4"
vr = VideoReader(video_fname, width=320, height=256)
duration = len(vr)
print('The video contains %d frames' % duration)

image = vr[1482].asnumpy()

final_answer = "Given "

inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

final_answer = final_answer + generated_text


prompt = "Question: Are people angry? Answer:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)

generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

is_angry = False
if generated_text.startswith("yes") or generated_text.startswith("Yes"):
    is_angry = True
    print("this is angry")
else:
    print("this is not angry")

prompt = "Question: Are people sad? Answer:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)

generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

is_sad = False
if generated_text.startswith("yes") or generated_text.startswith("Yes"):
    is_sad = True
    print("this is sad")
else:
    print("this is not sad")

prompt = "Question: Are people happy? Answer:"
inputs = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)

generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(generated_text)

is_happy = False
if generated_text.startswith("yes") or generated_text.startswith("Yes"):
    is_happy = True
    print("this is happy")
else:
    print("this is not happy")

have_prev = False
if is_angry or is_sad or is_happy:
    final_answer = final_answer + ", and the theme is"
if is_angry:
    final_answer = final_answer + " angry"
    have_prev = True



if is_sad:
    if have_prev:
        final_answer = final_answer + " and"
    final_answer = final_answer + " sad"
    have_prev = True



if is_happy:
    if have_prev:
        final_answer = final_answer + " and"
    final_answer = final_answer + " happy"
    have_prev = True




final_answer = final_answer + ", generate an idea for music of it (in fifty words)"

# Convert the NumPy array to an image
image = Image.fromarray(image)

# Save the image with a filename
image.save("new_image.jpg")

print("========== this is the final answer =========")
print(final_answer)

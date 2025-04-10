import torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

from configs import CFG

print(torch.__version__)
print(torch.cuda.is_available())


image_path = "C:/Users/rdas6/OneDrive/Desktop/codespace/pdf-parser/output/pdf2img/Griffiths - Introduction to quantum mechanics/Page_0003.png"

preprocessor_cnf_path = CFG.LAYOUT_MODEL_PATH.joinpath(*["model_artifacts", "layout", "preprocessor_config.json"])
model_cnf_path = CFG.LAYOUT_MODEL_PATH.joinpath(*["model_artifacts", "layout", "config.json"])
artifact_path = CFG.LAYOUT_MODEL_PATH.joinpath(*["model_artifacts", "layout"])
preprocessor = RTDetrImageProcessor.from_json_file(preprocessor_cnf_path)
model = RTDetrForObjectDetection.from_pretrained(artifact_path, config=model_cnf_path).to("cpu")
resize_cng = {"height": 640, "width": 640}
images = Image.open(image_path)
inputs = preprocessor(images, return_tensors="pt", size=resize_cng).to("cpu")
outputs = model(**inputs)
# results = preprocessor.post_process_object_detection(outputs, target_sizes=images.size[::-1], threshold=0.4)

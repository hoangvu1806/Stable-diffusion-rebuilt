from transformers import BlipProcessor, BlipModel, AutoTokenizer
from torch.nn.functional import cosine_similarity

# Giả sử bạn đã có một prompt và đường dẫn đến cache_dir
prompt1 = "a robot sitting on top of a computer"
prompt2 = "a river with flowers and trees in the fore"
cache_dir = "E:/Stable Diffusion rebuild/saved_models"

# Khởi tạo BlipProcessor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=cache_dir)
# Chuẩn bị prompt (ở đây chỉ là một ví dụ đơn giản)
input1s = processor(text=[prompt1], return_tensors="pt", max_length=9)  # Adjust max_length to match the shorter prompt
input2s = processor(text=[prompt2], return_tensors="pt", max_length=9)  # Adjust max_length to match the shorter prompt

output = tokenizer.encode(prompt2, return_tensors='pt')
# Lấy vector embedding (bạn có thể cần điều chỉnh tùy theo cấu trúc đầu ra của BlipProcessor)
text1_embedding = input1s.input_ids.float()
text2_embedding = input2s.input_ids.float()
similar = cosine_similarity(text1_embedding, text2_embedding)
print(output)
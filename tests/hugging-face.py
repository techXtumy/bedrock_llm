from huggingface_hub import InferenceClient

client = InferenceClient(api_key="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")

messages = [
	{
		"role": "user",
		"content": [
			{
				"type": "text",
				"text": "I want to build a LLM using Pytorch."
			},
			# {
			# 	"type": "image_url",
			# 	"image_url": {
			# 		"url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
			# 	}
			# }
		]
	}
]

stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-72B-Instruct", 
	messages=messages, 
	max_tokens=500,
	stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end="", flush=True)
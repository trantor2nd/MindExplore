from transformers import DistilBertTokenizer, DistilBertModel

tokenizer = DistilBertTokenizer.from_pretrained('/home/imagelab/zys/checkpoint/distilbert-base-uncased')
model = DistilBertModel.from_pretrained("/home/imagelab/zys/checkpoint/distilbert-base-uncased")
text = ""
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output.last_hidden_state.shape)
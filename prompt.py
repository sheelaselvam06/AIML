import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType

# Load model and tokenizer
model_name = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Set padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Sample dataset (you should replace this with your actual dataset)
sample_texts = [
    "This product is amazing! I love it.",
    "Terrible quality, would not recommend.",
    "Okay product, nothing special.",
    "Excellent customer service and fast delivery.",
    "Poor packaging, item arrived damaged."
]

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples, truncation=True, padding=True, max_length=128)

tokenized = [tokenize_function(text) for text in sample_texts]

config3 = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
 
peft_model3=get_peft_model(model,config3)
 
training_args = TrainingArguments(
    output_dir="./falcon_peft_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=5e-4,
    logging_steps=10,
    fp16=True,
    save_strategy="no",
    report_to=[],
)
 
trainer = Trainer(
    model=peft_model3,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)
 
trainer.train()
 
peft_model.save_pretrained("falcon-peft-tuned")
 
def build_prompt(review:str)->str:
    return f""" #Instruction:
    Classify the sentiment of the review.
 
    ### Input:
    {review}
 
    ### Response:
    """
review = "The product quality was excellent but delivery was late."
prompt=build_prompt(review)
inputs=tokenizer(prompt,return_tensors="pt").to(model.device)
 
with torch.no_grad():
    output_ids=model.generate(
        **inputs,
        max_new_tokens=5,
        temperature=0.0,
        do_sample=False,
    )
text=tokenizer.decode(output_ids[0],skip_special_tokens=True)
prediction = text.split("Sentiment:")[-1].strip()
 
print("Prediction:",prediction)
 
 
 
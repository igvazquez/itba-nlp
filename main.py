import torch
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, get_scheduler, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset

biden = open('dataset/biden_tweets.txt').read().splitlines()
trump = open('dataset/trump_tweets.txt').read().splitlines()

seed = 40

train_data, test_data = train_test_split(trump, test_size=0.3, random_state=seed)

test_data, val_data = train_test_split(test_data, test_size=0.5, random_state=seed)

# Initialize the GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

tokenizer.add_special_tokens({
  "eos_token": "</s>",
  "bos_token": "<s>",
  "unk_token": "<unk>",
  "pad_token":  tokenizer.eos_token,
  "mask_token": "<mask>"
})

train_tokenized = tokenizer(train_data, padding="max_length", truncation=True, max_length=128)
val_tokenized = tokenizer(val_data, padding="max_length", truncation=True, max_length=128)
test_tokenized = tokenizer(test_data, padding="max_length", truncation=True, max_length=128)

class TweetDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, index):
        return {key: torch.tensor(val[index]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


train_dataset = TweetDataset(train_tokenized)
val_dataset = TweetDataset(val_tokenized)
test_dataset = TweetDataset(test_tokenized)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

max_epochs = 10
scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=max_epochs * len(train_loader))

# Filtered trump tweets: 879
# Filtered biden tweets: 439

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(max_epochs):
    model.train()
    train_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["input_ids"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        train_loss += loss.item()

        loss.backward()
        optimizer.step()
        # scheduler.step()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}: Average training loss = {avg_train_loss}")

    # Evaluate on the validation set
    model.eval()
    eval_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone().detach()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            eval_loss += loss.item()

    avg_eval_loss = eval_loss / len(val_loader)
    print(f"Epoch {epoch + 1}: Average validation loss = {avg_eval_loss}")

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone().detach()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test loss: {avg_test_loss}")

model.eval()
prompt = "Trump is "
max_length = 140

input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

output = model.generate(
    # input_ids=input_ids,
    max_length=max_length,
    pad_token_id=tokenizer.eos_token_id,
    do_sample=True,
    num_return_sequences=5,
    top_k=50,
    top_p=0.95,
    temperature=1.0,
)

generated_tweets = [tokenizer.decode(tweet, skip_special_tokens=True) for tweet in output]

# Print generated tweets
for tweet in generated_tweets:
    print(tweet)
import pandas as pd
from itertools import islice
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PATH_TO_MODEL = ""
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(PATH_TO_MODEL)

tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token

# # Load the evaluation data
TEST_DATA = ""
test_df = pd.read_pickle(TEST_DATA)

prefix = "emotion classifier"

# # Prepare the data for testing
to_predict = [
    prefix + ": " + str(input_text)
    for input_text in test_df["text"].tolist()
]

to_predict = iter(to_predict)
predict_list = []

while (batch_lines := tuple(islice(to_predict, 500))):

    inputs = tokenizer(list(batch_lines), return_tensors="pt", padding=True)

    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=2,
        do_sample=False,
    )

    predict_list.extend(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))

output = pd.DataFrame({
    'id': test_df['tweet_id'].tolist(),
    'emotion': predict_list
})

OUTPUT_FILE = ""
output.to_csv(OUTPUT_FILE, columns=['id', 'emotion'], index=False)
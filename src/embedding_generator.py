import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModel



import torch
import pandas as pd 
from tqdm import tqdm
from constants import BASE_DIR

def embedding_generator(df):
    """
    Generates embeddings for the concatenated text of the main_tweet and reply_tweet columns in the input DataFrame.
    """
    # Load the pre-trained model and tokenizer
    config = AutoConfig.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
    tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
    model = AutoModel.from_pretrained("HooshvareLab/bert-base-parsbert-uncased")
    # Concatenate "main_tweet" and "reply_tweet" columns
    df['concatenated_text'] =  "توییت :" + " " + df['main_tweet'] + " ریپلای : "+  df['reply_tweet']

    # Create an empty list to store the embeddings
    embeddings = []
    indexes = []
    concatenated_text = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        # Tokenize the text
        encoded_input = tokenizer(row["concatenated_text"], return_tensors='pt')
        with torch.no_grad():
            # Get the model outputs
            outputs = model(**encoded_input)

        # Get the pooled output
        pooled_output = outputs.pooler_output
        embeddings.append(pooled_output.squeeze().tolist())
        indexes.append(row["index"])
        concatenated_text.append(row["concatenated_text"])

    return pd.DataFrame({"index": indexes, "concatenated_text": concatenated_text, "embeddings": embeddings})


if __name__ == '__main__':
    # Load the data
    data_path = BASE_DIR / "data/FinalDataWithSwearDetectedAndStanceAndCleaned1402_10_24.csv"
    df = pd.read_csv(data_path)

    # Generate embeddings
    df_with_embedding = embedding_generator(df)

    pd.to_csv(BASE_DIR / "data/FinalDataWithSwearDetectedAndStanceAndCleaned1402_10_24_with_embeddings.csv", index=False)
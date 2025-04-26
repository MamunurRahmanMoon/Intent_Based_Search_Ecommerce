from datasets import load_dataset, Dataset, DatasetDict
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader
import pandas as pd
from typing import Dict


# Fine tuning a SentenceTransformer model on the WDC Products-2017 dataset
def load_wdc_dataset(
    subsets: list,
    train_size: int = 5000,
    val_size: int = 1000,
):
    """
    Load and preprocess data from multiple subsets of the WDC Products-2017 dataset.

    Args:
        subsets (list): List of subsets of the dataset to load.
        train_size (int): Number of training samples to use from each subset.
        val_size (int): Number of validation samples to use from each subset.

    Returns:
        train_examples (list): List of InputExample for training.
        val_examples (list): List of InputExample for validation.
    """
    train_examples = []
    val_examples = []

    for subset in subsets:
        dataset = load_dataset("wdc/products-2017", subset)

        # Extract relevant columns and limit the size
        train_data = dataset["train"].select(
            range(min(train_size, len(dataset["train"])))
        )
        val_data = dataset["validation"].select(
            range(min(val_size, len(dataset["validation"])))
        )

        def convert_to_input_examples(data):
            examples = []
            for row in data:
                text_left = (
                    (row["title_left"] or "") + " " + (row["description_left"] or "")
                )
                text_right = (
                    (row["title_right"] or "") + " " + (row["description_right"] or "")
                )

                examples.append(
                    InputExample(
                        texts=[text_left.strip(), text_right.strip()],
                        label=float(row["label"]),
                    )
                )
            return examples

        train_examples.extend(convert_to_input_examples(train_data))
        val_examples.extend(convert_to_input_examples(val_data))

    return train_examples, val_examples


# Fine-tune a SentenceTransformer model (all-MiniLM-L6-v2) using the WDC dataset
def fine_tune_model(
    train_examples,
    val_examples,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    output_dir="fine_tuned_model",
    epochs=1,
    batch_size=16,
):
    """
    Fine-tune a SentenceTransformer model on the WDC dataset using titles and descriptions.

    Args:
        train_examples (list): List of InputExample for training.
        val_examples (list): List of InputExample for validation.
        model_name (str): Pre-trained SentenceTransformer model to use.
        output_dir (str): Directory to save the fine-tuned model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    """
    # Load the pre-trained model
    model = SentenceTransformer(model_name)

    # Create DataLoaders for training and validation
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    val_dataloader = DataLoader(val_examples, shuffle=False, batch_size=batch_size)

    # Define the loss function
    train_loss = losses.CosineSimilarityLoss(model)

    # Fine-tune the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        evaluation_steps=100,
        output_path=output_dir,
    )

    print(f"Model fine-tuned and saved to {output_dir}")


def prepare_dataset(file_path: str, train_frac: float = 0.8) -> DatasetDict:
    """
    Load and preprocess the dataset for NER fine-tuning.

    Args:
        file_path (str): Path to the dataset file.
        train_frac (float): Fraction of data to use for training (default: 0.8).

    Returns:
        DatasetDict: A Hugging Face DatasetDict with train and validation splits.
    """
    import pandas as pd
    from datasets import Dataset

    # Load the dataset
    data = pd.read_csv(file_path)

    # Ensure required columns exist
    if not all(col in data.columns for col in ["text", "entities"]):
        raise ValueError("Dataset must contain 'text' and 'entities' columns.")

    # Split into train and validation sets
    train_data = data.sample(frac=train_frac, random_state=42)
    val_data = data.drop(train_data.index)

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)

    return DatasetDict({"train": train_dataset, "validation": val_dataset})


def fine_tune_model(
    dataset: DatasetDict,
    model_name: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
):
    """
    Fine-tune a pre-trained model on the given dataset.

    Args:
        dataset (DatasetDict): The dataset for training and validation.
        model_name (str): Pre-trained model name.
        output_dir (str): Directory to save the fine-tuned model.
        epochs (int): Number of training epochs (default: 3).
        batch_size (int): Batch size for training (default: 16).
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        TrainingArguments,
        Trainer,
    )

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(dataset["train"].features["ner_tags"].feature.names)
    )

    # Tokenize the dataset
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                else:
                    label_ids.append(label[word_id])
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(output_dir)
    print(f"Model fine-tuned and saved to {output_dir}")


if __name__ == "__main__":
    # Define all subsets
    all_subsets = [
        "cameras_small",
        "cameras_medium",
        "cameras_large",
        "cameras_xlarge",
        "computers_small",
        "computers_medium",
        "computers_large",
        "computers_xlarge",
        "shoes_small",
        "shoes_medium",
        "shoes_large",
        "shoes_xlarge",
        "watches_small",
        "watches_medium",
        "watches_large",
        "watches_xlarge",
    ]

    # Load dataset from all subsets
    train_examples, val_examples = load_wdc_dataset(subsets=all_subsets)

    # Fine-tune the model
    fine_tune_model(train_examples, val_examples)

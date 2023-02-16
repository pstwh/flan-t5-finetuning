# Credits to https://www.philschmid.de/fine-tune-flan-t5 for the amazing blog post

import argparse
import os

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import evaluate

import nltk

nltk.download("punkt", quiet=True)

from dataset import BaseDataset, DefaultDataset
from processing import DefaultPreprocessor, DefaultPostprocessor
from metrics import DefaultMetrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="flan-t5-finetunning", description="Basic flan-t5 finetunner"
    )

    parser.add_argument("csv_file", type=str)
    parser.add_argument("--model_id", type=str, default="google/flan-t5-base")
    parser.add_argument("--task", type=str, default="summarize")
    parser.add_argument("--special_tokens_file", type=str, default="special_tokens.txt")
    parser.add_argument("--source_column", type=str, default="source")
    parser.add_argument("--target_column", type=str, default="target")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--model_save_name", type=str, default="custom")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    if os.path.isfile(args.special_tokens_file):
        with open(args.special_tokens_file) as f:
            tags = f.readlines()
            tags = list(map(lambda x: x.strip(), tags))
            tokenizer.add_tokens(tags, special_tokens=True)

    base = BaseDataset(
        args.csv_file,
        source_column=args.source_column,
        target_column=args.target_column,
    )
    preprocessor = DefaultPreprocessor(tokenizer, task=args.task)
    postprocessor = DefaultPostprocessor()

    metric = evaluate.load("rouge")
    metrics = DefaultMetrics(metric, tokenizer, postprocessor)

    dataset = DefaultDataset(base, preprocessor)

    model_id = args.model_id
    repository_id = f"{model_id.split('/')[1]}-{args.model_save_name}"

    tokenized_dataset = dataset.tokenize(dataset, test_size=args.test_size)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.resize_token_embeddings(len(tokenizer))

    label_pad_token_id = -100

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_dir=f"{repository_id}/logs",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        # report_to="tensorboard",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=metrics.compute,
    )

    trainer.train()

    # Samples
    # from random import randrange
    # import torch

    # for _ in range(3):
    #     sample = tokenized_dataset["test"][randrange(len(tokenized_dataset["test"]))]

    #     input_ids = torch.tensor(sample["input_ids"])
    #     attention_mask = torch.tensor(sample["attention_mask"])

    #     print("-- RAW --")
    #     print(
    #         preprocessor.__tokenizer__()
    #         .decode(input_ids)
    #         .replace("<pad>", "")
    #         .replace("</s>", "")
    #         .strip()
    #     )
    #     print()

    #     input_ids = input_ids.cuda()

    #     response = model.generate(input_ids, max_length=512).cpu()
    #     print(response)

    #     decoded = (
    #         preprocessor.__tokenizer__()
    #         .decode(response)
    #         .replace("<pad>", "")
    #         .replace("</s>", "")
    #         .strip()
    #     )

    #     print("-- RESPONSE --")
    #     print(decoded)
    #     print()
    #     print("--" * 5)
    #     print()

import pandas as pd

from datasets import Dataset, concatenate_datasets


class BaseDataset:
    def __init__(self, csv_path, source_column, target_column):
        df = pd.read_csv(csv_path)
        df = df[[source_column, target_column]]
        df = df.rename(columns={source_column: "source", target_column: "target"})

        self.df = df

    def dataframe(self):
        return self.df


class DefaultDataset:
    def __init__(self, dataset, preprocessor):
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.source_column = "source"
        self.target_column = "target"
        self.remove_columns = [self.source_column, self.target_column]

    def tokenize(self, tags=[], test_size=0.1):
        dataset = Dataset.from_pandas(self.dataset.dataframe())
        dataset = dataset.train_test_split(test_size=test_size)

        tokenized_inputs = concatenate_datasets(
            [dataset["train"], dataset["test"]]
        ).map(
            lambda x: self.preprocessor.__tokenizer__()(
                x[self.source_column], truncation=True
            ),
            batched=True,
            remove_columns=self.remove_columns,
        )
        max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])

        tokenized_targets = concatenate_datasets(
            [dataset["train"], dataset["test"]]
        ).map(
            lambda x: self.preprocessor.__tokenizer__()(
                x[self.target_column], truncation=True
            ),
            batched=True,
            remove_columns=self.remove_columns,
        )
        max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])

        tokenized_dataset = dataset.map(
            lambda sample: self.preprocessor.process(
                sample, max_target_length, max_source_length
            ),
            batched=True,
            remove_columns=self.remove_columns,
        )

        return tokenized_dataset

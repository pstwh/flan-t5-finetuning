from nltk.tokenize import sent_tokenize


class DefaultPreprocessor:
    def __init__(
        self,
        tokenizer,
        padding="max_length",
        task="summarize",
        source_column="source",
        target_column="target",
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.source_column = source_column
        self.target_column = target_column
        self.task = task

    def __tokenizer__(self):
        return self.tokenizer

    def process(self, sample, max_target_length=512, max_source_length=512):
        inputs = [self.task + ": " + item for item in sample[self.source_column]]

        model_inputs = self.tokenizer(
            inputs,
            max_length=max_source_length,
            padding=self.padding,
            truncation=True,
        )

        labels = self.tokenizer(
            sample[self.target_column],
            max_length=max_target_length,
            padding=self.padding,
            truncation=True,
        )

        if self.padding == "max_length":
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs


class DefaultPostprocessor:
    def process(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(sent_tokenize(label)) for label in labels]

        return preds, labels

import src.functional as functional
from collections import defaultdict

def get_soft_labels(datasets, mt, device):
    """Obtain soft labels from a masked language model for all samples in the datasets."""

    soft_labels = defaultdict()

    for dataset in datasets:
        for _, _, relation in dataset:
            for sample in relation.samples:
                prompt = functional.make_prompt(
                    mt=mt, prompt_template=relation.prompt_templates[0], subject=sample.subject
                )
                _h_index, inputs = functional.find_subject_token_index(
                    mt=mt, prompt=prompt, subject=sample.subject
                )
                inputs = mt.tokenizer(prompt, return_tensors="pt").to(mt.model.device)
                input_ids = inputs.input_ids

                outputs = mt.model(input_ids=input_ids[:, :_h_index], use_cache=True)

                if relation.name not in soft_labels:
                    soft_labels[relation.name] = {}

                soft_labels[relation.name][sample.subject] = outputs.logits
    return soft_labels

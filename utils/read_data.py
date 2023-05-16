import json

from torch.utils.data import Dataset


class UIDataset(Dataset):
    def __init__(self, jsonl_file):
        self.dataset = [instance for instance in self.preprocess_jsonl(jsonl_file)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]

    def preprocess_jsonl(self, jsonl_file: str):
        for line in open(jsonl_file):
            sample = json.loads(line)
            try:
                for instance in sample['instances']:
                    input_ = 'input: ' \
                            + instance['instruction_with_input'].strip() \
                            + '\n' \
                            + instance['constraints'].strip() 
                    yield input_, instance['output'].strip()
            except:
                input_ = 'input: ' \
                        + sample['instruction'].strip() \
                        + '\n' \
                        + sample['constraints'].strip() \
                        + '\n' \
                        + sample['input'].strip() \
                        + '\n' \
                        + 'Output:'
                yield input_, sample['output'].strip()


# Create init main function
if __name__ == "__main__":
    print("read data")
    output = read_en_wic("data/en_wic/val.jsonl")
    for o in output[0][0].split('\n'):
        print(o)
from torch.utils.data import Dataset

            
class SSPDataset(Dataset):
    def __init__(self, tokenizer, filepath, max_length): 
        sentence_pairs = []
        labels = []
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                splitline = line.split('\x03')
                sentence_pairs.append([splitline[0], splitline[1]])
                labels.append(int(splitline[2]))
                
        self.batch_encoding = tokenizer(sentence_pairs, add_special_tokens=True, truncation=True, max_length=max_length)
        self.labels = labels
        
    def __getitem__(self, idx):
        return {
            'input_ids': self.batch_encoding['input_ids'][idx],
            'attention_mask': self.batch_encoding['attention_mask'][idx],
            'token_type_ids': self.batch_encoding['token_type_ids'][idx],
            'next_sentence_label': self.labels[idx]
        }
    
    def __len__(self):
        return len(self.labels)
        
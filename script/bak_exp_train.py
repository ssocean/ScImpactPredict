   class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = float(row['TNCSI'])
        if args.data_style == 0:

            text = f"Given a certain paper, Title: {row['title']}\n Abstract: {row['abstract']}. \n Predict its normalized scholar impact (between 0 and 1):"
            
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        if args.data_style == 1:    
            most_cite = int(row['authors_title'])
            oa = row['OA']
            if oa is None or oa == 'None':
                oa = '\n The code of the paper has been shared on the web. \n'
            else:
                oa = '\n The code of the paper is unavailable. \n'
            if most_cite >10000:
                authors_title = '\n This article was written by a renowned scholar in the field. \n'
            else:
                authors_title = '\n The author of this article is not very well-known in their field. \n'
            text = f"Given a certain paper entitled {row['title']}, predict its normalized scholar impact (between 0 and 1).{oa}{authors_title} The Abstract of paper: {row['abstract']}."
            
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 2:

            most_cite = int(row['authors_title'])
            oa = row['OA']
            if oa is None or oa == 'None':
                oa = '\n The code of the paper has been shared on the web. \n'
            else:
                oa = '\n The code of the paper is unavailable. \n'
                
            text = f"Given a certain paper entitled {row['title']}, predict its normalized scholar impact (between 0 and 1).{oa} The Abstract of paper: {row['abstract']}."
            
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 3:

            most_cite = int(row['authors_title'])
            oa = row['OA']
            if oa is None or oa == 'None':
                oa = 'NO'
            else:
                oa = 'YES'
            text = f'''
            Title: "{row['title']}"
            Abstract: "{row['abstract']}"
            Open Access: {oa}

            Please predict the normalized scholarly impact of the described paper, with a value between 0 and 1, where 0 represents minimal impact and 1 represents maximum impact. Base your prediction on the relevance, novelty, and potential influence of the research described.
            '''
            
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 4:
            most_cite = int(row['authors_title'])
            oa = row['OA']
            if oa is None or oa == 'None':
                oa = 'NO'
            else:
                oa = 'YES'
            # 分类作者最高引用次数
            if most_cite <= 500:
                citation_category = '\nThis article was written by an author with a low citation count.\n'
            elif most_cite <= 500:
                citation_category = '\nThis article was written by an author with a low citation count.\n'
            elif most_cite <= 1000:
                citation_category = '\nThis article was written by an author with a medium citation count.\n'
            elif most_cite <= 5000:
                citation_category = '\nThis article was written by an author with a high citation count.\n'
            else:
                citation_category = '\nThis article was written by an author with a very high citation count.\n'
                
            text = f'''
            Title: "{row['title']}"
            Abstract: "{row['abstract']}"
            Open Access: {oa}
            {citation_category}
            Please predict the normalized scholarly impact of the described paper, with a value between 0 and 1, where 0 represents minimal impact and 1 represents maximum impact. Base your prediction on the relevance, novelty, and potential influence of the research described.
            '''
            
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 5:
            most_cite = int(row['authors_title'])
            oa = row['OA']
            if oa is None or oa == 'None':
                oa = 'NO'
            else:
                oa = 'YES'
            # 分类作者最高引用次数
            if most_cite <= 1000:
                citation_category = '\nThis article was written by an author with a low citation count.\n'
            elif most_cite <= 5000:
                citation_category = '\nThis article was written by an author with a high citation count.\n'
            else:
                citation_category = '\nThis article was written by an author with a very high citation count.\n'
                
            text = f'''
            Title: "{row['title']}"
            Abstract: "{row['abstract']}"
            Open Access: {oa}
            {citation_category}
            Please predict the normalized scholarly impact of the described paper, with a value between 0 and 1, where 0 represents minimal impact and 1 represents maximum impact. Base your prediction on the relevance, novelty, and potential influence of the research described.
            '''
            
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
class TextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        label = float(row['TNCSI_SP'])
        if args.data_style == 0:
            text = f'''Given a certain paper, Title: {row['title']}\n Abstract: {row['abstract']}. \n Predict its normalized academic impact (between 0 and 1):'''
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 1:

            most_cite = int(row['authors_title'])
            oa = row['OA']
            if oa is None or oa == 'None':
                oa = '\n The code of the paper has been shared on the web. \n'
            else:
                oa = '\n The code of the paper is unavailable. \n'
            if most_cite >10000:
                authors_title = '\n This article was written by a renowned scholar in the field. \n'
            else:
                authors_title = '\n The author of this article is not very well-known in their field. \n'
            text = f"Given a certain paper entitled {row['title']}, predict its normalized scholar impact (between 0 and 1).{oa}{authors_title} The Abstract of paper: {row['abstract']}."
            
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 2:

            most_cite = int(row['authors_title'])
            oa = row['OA']
            if oa is None or oa == 'None':
                oa = '\n The code of the paper has been shared on the web. \n'
            else:
                oa = '\n The code of the paper is unavailable. \n'
                
            text = f"Given a certain paper entitled {row['title']}, predict its normalized scholar impact (between 0 and 1).{oa} The Abstract of paper: {row['abstract']}."
            
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 3:

            most_cite = int(row['authors_title'])
            oa = row['OA']
            if oa is None or oa == 'None':
                oa = 'NO'
            else:
                oa = 'YES'
            text = f'''
            Title: "{row['title']}"
            Abstract: "{row['abstract']}"
            Open Access: {oa}

            Please predict the normalized scholarly impact of the described paper, with a value between 0 and 1, where 0 represents minimal impact and 1 represents maximum impact. Base your prediction on the relevance, novelty, and potential influence of the research described.
            '''
            
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 4:
            most_cite = int(row['authors_title'])
            oa = row['OA']
            if oa is None or oa == 'None':
                oa = 'NO'
            else:
                oa = 'YES'
            # 分类作者最高引用次数
            if most_cite <= 500:
                citation_category = '\nThis article was written by an author with a low citation count.\n'
            elif most_cite <= 500:
                citation_category = '\nThis article was written by an author with a low citation count.\n'
            elif most_cite <= 1000:
                citation_category = '\nThis article was written by an author with a medium citation count.\n'
            elif most_cite <= 5000:
                citation_category = '\nThis article was written by an author with a high citation count.\n'
            else:
                citation_category = '\nThis article was written by an author with a very high citation count.\n'
                
            text = f'''
            Title: "{row['title']}"
            Abstract: "{row['abstract']}"
            Open Access: {oa}
            {citation_category}
            Please predict the normalized scholarly impact of the described paper, with a value between 0 and 1, where 0 represents minimal impact and 1 represents maximum impact. Base your prediction on the relevance, novelty, and potential influence of the research described.
            '''
            
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 5:
            most_cite = int(row['authors_title'])
            oa = row['OA']
            if oa is None or oa == 'None':
                oa = 'NO'
            else:
                oa = 'YES'
            # 分类作者最高引用次数
            if most_cite <= 1000:
                citation_category = '\nThis article was written by an author with a low citation count.\n'
            elif most_cite <= 5000:
                citation_category = '\nThis article was written by an author with a high citation count.\n'
            else:
                citation_category = '\nThis article was written by an author with a very high citation count.\n'
                
            text = f'''
            Title: "{row['title']}"
            Abstract: "{row['abstract']}"
            Open Access: {oa}
            {citation_category}
            Please predict the normalized scholarly impact of the described paper, with a value between 0 and 1, where 0 represents minimal impact and 1 represents maximum impact. Base your prediction on the relevance, novelty, and potential influence of the research described.
            '''
            
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 6:
            most_cite = int(row['authors_title'])
            oa = row['OA']
            if oa is None or oa == 'None':
                oa = 'NO'
            else:
                oa = 'YES'
            # 分类作者最高引用次数
            if most_cite <= 1000:
                citation_category = '\nThis article was written by an author with a low citation count.\n'
            elif most_cite <= 5000:
                citation_category = '\nThis article was written by an author with a high citation count.\n'
            else:
                citation_category = '\nThis article was written by an author with a very high citation count.\n'
                
            text = f'''
            Title: "{row['title']}"
            Abstract: "{row['abstract']}"
            Please predict the normalized scholarly impact of the described paper, with a value between 0 and 1, where 0 represents minimal impact and 1 represents maximum impact. Base your prediction on the relevance, novelty, and potential influence of the research described.
            '''
            
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 7:
            most_cite = int(row['authors_title'])
            oa = row['OA']
            citation_category = '\nThis article was written by an author with a low citation count.\n'
            oa = 'NO'
            text = f'''
            Title: "{row['title']}"
            Abstract: "{row['abstract']}"
            Open Access: {oa}
            {citation_category}
            Please predict the normalized scholarly impact of the described paper, with a value between 0 and 1, where 0 represents minimal impact and 1 represents maximum impact. Base your prediction on the relevance, novelty, and potential influence of the research described.
            '''
            
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 8:
            most_cite = int(row['authors_title'])
            oa = row['OA']
            citation_category = '\nThis article was written by an author with unknown citation count.\n'
            oa = 'Unkown'
            text = f'''
            Title: "{row['title']}"
            Abstract: "{row['abstract']}"
            Open Access: {oa}
            {citation_category}
            Please predict the normalized scholarly impact of the described paper, with a value between 0 and 1, where 0 represents minimal impact and 1 represents maximum impact. Base your prediction on the relevance, novelty, and potential influence of the research described.
            '''
            
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 9:
            most_cite = int(row['authors_title'])
            oa = row['OA']
            citation_category = '\nThis article was written by an author with a very high citation count.\n'
            oa = 'Yes'
            text = f'''
            Title: "{row['title']}"
            Abstract: "{row['abstract']}"
            Open Access: {oa}
            {citation_category}
            Please predict the normalized scholarly impact of the described paper, with a value between 0 and 1, where 0 represents minimal impact and 1 represents maximum impact. Base your prediction on the relevance, novelty, and potential influence of the research described.
            '''
            
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 66:
            text = f'''Title: {row['title']}\n Abstract: {row['abstract']}'''
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        elif args.data_style == 88:
            text = f'''Given a certain paper, Title: {row['title']}\n Abstract: {row['abstract']}. \n Predict its normalized academic impact (between 0 and 1):'''
            inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True,
                                    return_tensors="pt")
            return {
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.float)
            }
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from typing import Any 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

class Dataset():
    def __init__(self, data_path: str, test_size: float = 0.2, random_state: int = 42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.input, self.output = self.load()

    def load(self) -> Any:
        population_params = []     
        examples = []        
        if osp.exists(osp.join(self.data_path, 'dataset.npz')):
            dataset = np.load(osp.join(self.data_path, 'dataset.npz'))   

            return dataset['examples'], dataset['population_params']

        else:                                                
            directories = os.listdir(self.data_path)                                      
            for idx, example in enumerate(directories[:15]):
                example_path = osp.join(self.data_path, example)

                params = pd.read_csv(osp.join(example_path, 'configuration.csv'))
                population_params.append(params.values[0])

                events = []
                for event in tqdm(os.listdir(example_path), desc=f'Loading events for {example}, {idx+1}'):
                    if event == 'configuration.csv':
                        continue
                    event_path = osp.join(example_path, event)

                    data = pd.read_csv(event_path, delim_whitespace=True, comment='#', header=None, names=['m1_source', 'm2_source'])
                    means = np.mean(data.values, axis=0)
                    events.append(means)
                examples.append(events)
            examples = np.array(examples)
            examples = examples.reshape(examples.shape[0], -1)

            np.savez(osp.join(self.data_path, 'dataset.npz'), examples=examples, population_params=population_params)

        return examples, population_params
    
    def get_dims(self):
        return self.input.shape[1], self.output.shape[1]
    
    def get_dataset(self):
        X_train, X_test, y_train, y_test = train_test_split(self.input, self.output, test_size=self.test_size, random_state=self.random_state)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


        return [X_train, y_train], [X_test, y_test]
    
def get_dataloader(trainset, testset, val_size, batch_size):
    X_train, y_train = trainset
    X_test, y_test = testset
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=42)

    X_train, y_train, X_val, y_val, X_test, y_test = map(lambda x: torch.tensor(x, dtype=torch.float32),
                                                         (X_train, y_train, X_val, y_val, X_test, y_test))


    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(list(zip(X_val, y_val)), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(list(zip(X_test, y_test)), shuffle=True)

    return train_loader, valid_loader, test_loader

if __name__ == '__main__':
    dataset = Dataset(data_path='/home/safi/Semester 07/Kaavish/bnn/data')
    dataset.load()
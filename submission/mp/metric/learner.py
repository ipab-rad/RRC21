
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, dataset
from pytorch_metric_learning import losses, miners, distances, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from utils import GraspMetricDataset

class Embedder(nn.Module):

    def __init__(self):

        self.net = nn.Sequential(
            nn.Linear(in_features=3, out_features=10),
            nn.ReLU(),
            nn.Linear(in_features=10,out_features=5),
            nn.Dropout(p=0.5),
            nn.ReLU(),
        )

    def forward(self, x):

        return self.net(x)


def main(args):

    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low = 0)
    criterion = losses.TripletMarginLoss(margin = 0.2,
                                        distance = distance,
                                        reducer = reducer)
    miner = miners.TripletMarginMiner(margin = 0.2,
                                    distance = distance,
                                    type_of_triplets = "semihard")
    tester = testers.BaseTester()
    evaluator = AccuracyCalculator(k = 1)

    model = Embedder.to(args.device)

    train_dataset = dataset.MNIST(args.train_path, train=True, download=True, )
    test_dataset = dataset.MNIST(args.test_path, train=False, download=True, )
    train_loader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True)

    optimizer = torch.optim.Adam(model,lr=0.00001, weight_decay=0.0001)

    for epoch in range(1, args.epochs+1):

        ## training loop
        for idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            emb = model(data)
            anchors = miner(emb, labels)
            loss = criterion(emb, labels, anchors)
            loss.backward()
            optimizer.step()
            if idx % args.freq == 0:
                print(f'Epoch:{epoch} Iter:{idx} Loss:{loss} anchors:{anchors}')
        
        ## evaluation
        train_emb, train_labels = tester.get_all_embeddings(train_dataset, model)
        test_emb, test_labels = tester.get_all_embeddings(test_dataset, model)

        eval = evaluator.get_accuracy(test_emb,
                                    train_emb,
                                    test_labels,
                                    train_labels,
                                    False)
        precision = eval["precision_at_1"]

        print(f"Test (Precision@1):{precision}")

    
    torch.save(model.state_dict(), args.save_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=32)
    parser.add_argument('--epochs', default=4)
    parser.add_argument('--freq', default=20)
    parser.add_argument('--train-path', default=".")
    parser.add_argument('--test-path', default=".")
    parser.add_argument('--save-path', default='./grasp_metric.pt')

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(args)
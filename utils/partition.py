import torch
import itertools


class Partition:
    def __init__(self, size):
        self.classes = torch.zeros(8)  # 8 is the number of classes
        self.partition = torch.ones(size) * -1
        self.scores = torch.zeros(8)
        self.score = self.scores.sum()

    def check_span(self, start, end):
        return self.partition[start : end + 1].sum() == start - end - 1

    def insert_span(self, start, end, label, score):
        if start != 0 and end != 0:
            self.partition[start : end + 1] = label

        self.classes[label] = 1
        self.scores[label] = score
        self.score += score

    def can_merge(self, partition):
        return (
            not torch.logical_and(self.classes, partition.classes).any()
        ) and not torch.logical_and(
            self.partition != -1, partition.partition != -1
        ).any()

    def merge_partitions(self, partition):
        self.partition = torch.where(
            self.partition == -1, partition.partition, self.partition
        )
        self.classes = torch.logical_or(self.classes, partition.classes)

    def update_score(self):
        self.score = self.scores.sum()

    def __repr__(self) -> str:
        return self.partition.__repr__()

    def __str__(self) -> str:
        return self.partition.__str__()


def partition_predictions(sep_index, starts, ends, target_spans, threshold=0, top_k=3):
    targ = target_spans[0]
    a = [
        [int(z) for z in x.argsort(descending=True) if x[z] > threshold][:top_k]
        for x in starts[:, :sep_index]
    ]
    b = [
        [int(z) for z in x.argsort(descending=True) if x[z] > threshold][:top_k]
        for x in ends[:, :sep_index]
    ]

    # get all combinations of starts and ends for each class
    all_combinations = [list(itertools.product(a[i], b[i])) for i in range(len(a))]
    valid_combos = [
        [
            z
            for z in x
            if z[0] <= z[1]
            and z[0] != 0
            and z[1] != 0
            and (
                (z[0] < targ[0] and z[1] < targ[0])
                or (z[0] > targ[1] and z[1] > targ[1])
            )
        ]
        + [(0, 0)]
        for x in all_combinations
    ]

    # sort the combinations by score
    sorted_combos = [
        sorted(x, key=lambda y: starts[i, y[0]] + ends[i, y[1]], reverse=True)
        for i, x in enumerate(valid_combos)
    ]

    # get each combination of each class
    possible_partitions = list(itertools.product(*sorted_combos))

    best_partition = (None, float("-inf"))

    for p in possible_partitions:
        partition = Partition(sep_index)
        for i, span in enumerate(p):
            if partition.check_span(span[0], span[1]):
                partition.insert_span(
                    span[0], span[1], i, starts[i, span[0]] + ends[i, span[1]]
                )
            else:
                break
        else:
            if partition.score > best_partition[1]:
                best_partition = (partition, partition.score)

    return best_partition[0]

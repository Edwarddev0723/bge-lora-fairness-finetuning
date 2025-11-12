class DataLoader:
    def __init__(self, dataset, batch_size, educational_background_col):
        self.dataset = dataset
        self.batch_size = batch_size
        self.educational_background_col = educational_background_col
        self.batches = self.create_batches()

    def create_batches(self):
        # Group samples by educational background
        grouped_data = {}
        for sample in self.dataset:
            background = sample[self.educational_background_col]
            if background not in grouped_data:
                grouped_data[background] = []
            grouped_data[background].append(sample)

        # Balance samples across educational backgrounds
        min_group_size = min(len(group) for group in grouped_data.values())
        balanced_batches = []
        
        for background, samples in grouped_data.items():
            # Shuffle samples for randomness
            np.random.shuffle(samples)
            # Take only the minimum number of samples from each group
            balanced_batches.extend(samples[:min_group_size])

        # Shuffle the balanced dataset
        np.random.shuffle(balanced_batches)

        # Create batches
        return [balanced_batches[i:i + self.batch_size] for i in range(0, len(balanced_batches), self.batch_size)]

    def get_batches(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
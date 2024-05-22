def train_test_split(data: np.array, train_portion = 0.7):
    # Make a copy from data
    data_copy = data.copy()
    
    # Shuffle the data
    np.random.shuffle(data_copy)

    # Calculate an edge for splitting
    edge = int(data.shape[0]*train_portion)

    # Return train and test portions respectively
    return data_copy[:edge], data_copy[edge:]
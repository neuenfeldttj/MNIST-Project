from mnist import *

def main():
    batch_size = 32
    # Load dataset
    print("Loading training and test data...")
    train_data, val_data = load_data("train-images-idx3-ubyte", "train-labels-idx1-ubyte", batch_size, train=True)
    print("Success!")

    # Train model
    print("Training model...")

    model = MNISTModel()
    criterion = torch.nn.CrossEntropyLoss() # best type of loss for multiclass
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001) # could use typical SGD, but more efficient

    num_epochs = 20

    #Create plot for visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    plt.suptitle("MNIST Training")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUROC")

    stats = [] # Array to store info from each epoch to plot

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}\n")
        # Train the model through forward prop and back prop
        model.train()
        for image, label in train_data:
            optimizer.zero_grad()
            predicted = model(image)
            loss = criterion(predicted, label)
            loss.backward()
            optimizer.step()

        # Get stats from current epoch
        get_train_val_stats(train_data, val_data, model, criterion, stats)

        # Update the graph
        update_graph(axes, epoch, stats)


        # Save current epoch params in case it's the one we want
        save_checkpoint(model, epoch, stats)

    plt.savefig("training.png", dpi=200)

    print("Finished Training!\n")


if __name__ == "__main__":
    main()

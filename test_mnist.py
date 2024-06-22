from mnist import *

def main():
    batch_size = 32

    print("Loading data...")
    test_data = load_data("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", batch_size)
    print("Success!")

    epoch = int(input("Enter an epoch to load in from: "))

    model, stats = get_checkpoint(epoch)
    criterion = nn.CrossEntropyLoss()

    print("Test Metrics")
    calc_metrics(test_data, model, criterion)

    save_checkpoint(model, epoch, [], "final_model.pt")

if __name__ == "__main__":
    main()
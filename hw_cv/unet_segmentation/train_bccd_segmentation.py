if __name__ == '__main__':
    import argparse
    from datetime import datetime
    import torch
    import utils
    from data_setup.bccd_data_setup import create_dataloaders
    from engine.model_builder import UNet
    from engine.engine import train
    from utils.unet_utils import unet_loss

    # Create a parser
    parser = argparse.ArgumentParser(description="Get some hyperparameters.")

    # Get an arg for num_epochs
    parser.add_argument("--num_epochs",
                        default=20,
                        type=int,
                        help="the number of epochs to train for")

    # Get an arg for batch_size
    parser.add_argument("--batch_size",
                        default=10,
                        type=int,
                        help="number of samples per batch")

    # Get an arg for learning_rate
    parser.add_argument("--learning_rate",
                        default=1e-3,
                        type=float,
                        help="learning rate to use for model")

    # BCCD dataset location
    parser.add_argument("--bccd_dataset_location",
                        default='/home/alex/projects/ml/ml_hw/hw_cv/bccd-dataset/',
                        type=str,
                        help="validation bccd-dataset size")

    # Get our arguments from the parser
    args = parser.parse_args()

    # Setup hyperparameters
    NUM_EPOCHS = args.num_epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    BCCD_DATASET_LOCATION = args.bccd_dataset_location
    print(
        f"[INFO] Training a model for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE} and a learning rate of {LEARNING_RATE}")

    # Setup target device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trn_dl, val_dl = create_dataloaders(BATCH_SIZE, BCCD_DATASET_LOCATION, device)

    model = UNet().to(device)
    criterion = unet_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    n_epochs = NUM_EPOCHS

    train(model, trn_dl, val_dl, criterion, optimizer, n_epochs)

    # Save the model with help from utils.py
    utils.save_model(model=model,
                     target_dir="models",
                     model_name="bccd_segm_model_{}.pth".format(datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')))

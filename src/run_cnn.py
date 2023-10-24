from classify import Cancer_Dataset, split_data, run_cnn
from torchvision import transforms
import glob

file_path = "../Cancerous cell smears"
paths = glob.glob(f"{file_path}/*")

transform = transforms.Compose(
    [transforms.ToTensor(),
        transforms.Normalize(mean=[0.406], std=[0.225])]
)
data_loader = Cancer_Dataset(file_path, transform=transform)
train_data, test_data = split_data(data_loader)
training_losses, training_accs, testing_losses, testing_accs = run_cnn(train_data, test_data, n_epochs=30)
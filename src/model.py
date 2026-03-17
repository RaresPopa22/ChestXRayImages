import torch.nn as nn

class BaseCNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        hyperparam_config = config['hyperparameters']
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(hyperparam_config['layer_1_dropout']),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same', stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(hyperparam_config['layer_2_dropout']),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.layer_3_a = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same', stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(hyperparam_config['layer_3_dropout']),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        self.layer_3_b = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same', stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(hyperparam_config['layer_3_dropout']),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.layer_3_c = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding='same', stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(hyperparam_config['layer_3_dropout']),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )

        self.fc1 = nn.Linear(512, 512)

        self.layer_4 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(hyperparam_config['layer_4_dropout'])
        )

        self.fc2 = nn.Linear(512, 1)

        self.layer_5 = nn.Sequential(
            self.fc2
        )

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3_a(out)
        out = self.layer_3_b(out)
        out = self.layer_3_c(out)
        out = out.mean(dim=[2, 3])
        out = self.layer_4(out)
        return self.layer_5(out)
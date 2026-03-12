import torch.nn as nn

class BaseCNN(nn.Module):

    def __init__(self, target_size):
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding='same', stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(32),
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same', stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(64)
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same', stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128)
        )

        self.fc1 = nn.Linear(128, 128)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.layer_4 = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Linear(128, 1)

        self.layer_5 = nn.Sequential(
            self.fc2
        )
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = out.mean(dim=[2, 3])
        out = self.layer_4(out)
        return self.layer_5(out)
    


def get_layer_with_relu(in_channel, out_channel, kernel_size=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channel, 
            out_channels=out_channel, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            stride=stride
            ),
        nn.BatchNorm2d(out_channel),
        nn.ReLU()
    )


def get_layer(in_channel, out_channel, kernel_size=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channel, 
            out_channels=out_channel, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            stride=stride
            ),
        nn.BatchNorm2d(out_channel)
    )


class ConvolutionalBlock(nn.Module):

    def __init__(self, in_channel, channels, kernel_size, stride):
        super().__init__()
        c1, c2, c3 = channels
        self.layer_1 = get_layer_with_relu(in_channel, c1, stride=stride)
        self.layer_2 = get_layer_with_relu(c1, c2, kernel_size)
        self.layer_3 = get_layer(c2, c3)
        self.layer_4 = get_layer(in_channel, c3, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_shortcut = x
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        x_shortcut = self.layer_4(x_shortcut)
        out = out + x_shortcut
        return self.relu(out)
    

class IdentityBlock(nn.Module):
    
    def __init__(self, in_channel, channels, kernel_size):
        super().__init__()
        c1, c2, c3 = channels
        self.layer_1 = get_layer_with_relu(in_channel, c1)
        self.layer_2 = get_layer_with_relu(c1, c2, kernel_size=kernel_size)
        self.layer_3 = get_layer(c2, c3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_shortcut = x
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = out + x_shortcut
        return self.relu(out)


class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer_0 = nn.ZeroPad2d(3)
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2)
        )

        self.conv_block_1 = ConvolutionalBlock(64, (64, 64, 256), 3, 1)
        self.identity_block_1 = IdentityBlock(256, (64, 64, 256), 3)
        self.identity_block_2 = IdentityBlock(256, (64, 64, 256), 3)
        
        self.conv_block_2 = ConvolutionalBlock(256, (128, 128, 512), 3, 2)

        self.identity_block_3 = IdentityBlock(512, (128, 128, 512), 3)
        self.identity_block_4 = IdentityBlock(512, (128, 128, 512), 3)
        self.identity_block_5 = IdentityBlock(512, (128, 128, 512), 3)
        
        self.conv_block_3 = ConvolutionalBlock(512, (256, 256, 1024), 3, 2)
        
        self.identity_block_6 = IdentityBlock(1024, (256, 256, 1024), 3)
        self.identity_block_7 = IdentityBlock(1024, (256, 256, 1024), 3)
        self.identity_block_8 = IdentityBlock(1024, (256, 256, 1024), 3)
        self.identity_block_9 = IdentityBlock(1024, (256, 256, 1024), 3)
        self.identity_block_10 = IdentityBlock(1024, (256, 256, 1024), 3)

        self.conv_block_4 = ConvolutionalBlock(1024, (512, 512, 2048), 3, 2)
        
        self.identity_block_11 = IdentityBlock(2048, (512, 512, 2048), 3)
        self.identity_block_12 = IdentityBlock(2048, (512, 512, 2048), 3)

        self.fc = nn.Linear(2048, 1)

    def forward(self, x):
        out = self.layer_0(x)
        out = self.layer_1(out)
        
        out = self.conv_block_1(out)
        out = self.identity_block_1(out)
        out = self.identity_block_2(out)

        out = self.conv_block_2(out)

        out = self.identity_block_3(out)
        out = self.identity_block_4(out)
        out = self.identity_block_5(out)

        out = self.conv_block_3(out)

        out = self.identity_block_6(out)
        out = self.identity_block_7(out)
        out = self.identity_block_8(out)
        out = self.identity_block_9(out)
        out = self.identity_block_10(out)

        out = self.conv_block_4(out)

        out = self.identity_block_11(out)
        out = self.identity_block_12(out)

        out = out.mean(dim=[2, 3])
        out = self.fc(out)
        return out

import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_generated_images(data_loader, model,
                          device, figsize=(20, 3),
                          n_images=15, model_type='autoencoder'):
    fig, axes = plt.subplots(2, n_images, sharex=True, sharey=True, figsize=figsize)

    for batch_id, (features, _) in enumerate(data_loader):
        features = features.to(device)

        color_channels = features.shape[1]
        image_height = features.shape[2]
        image_width = features.shape[3]

        with torch.no_grad():
            if model_type == 'autoencoder':
                decoded_images = model(features)[:n_images]
            else:
                raise ValueError('model type not supported')

        original_images = features[:n_images]
        break

    for i in range(n_images):
        for ax, img in zip(axes, [original_images, decoded_images]):
            current_image = img[i].detach().to(torch.device('cpu'))

            if color_channels > 1:
                current_image = np.transpose(current_image, (1, 2, 0))
                ax[i].imshow(current_image)
            else:
                ax[i].imshow(current_image.view((image_height, image_width)), cmap='binary')


def plot_latent_space_with_labels(num_classes, data_loader, model, device):
    d = {i: [] for i in range(num_classes)}

    model.eval()

    with torch.no_grad():
        for i, (features, targets) in enumerate(data_loader):

            features = features.to(device)
            targets = targets.to(device)

            embedding = model.encoder(features)

            for i in range(num_classes):
                if i in targets:
                    mask = targets == i
                    d[i].append(embedding[mask].to('cpu').numpy())

    colors = list(mcolors.TABLEAU_COLORS.items())
    for i in range(num_classes):
        d[i] = np.concatenate(d[i])
        plt.scatter(d[i][:, 0], d[i][:, 1], color=colors[i][1], label=f'{i}', alpha=0.5)

    plt.legend()
    plt.tight_layout()

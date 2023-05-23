from tqdm import tqdm


def train_fn(data_loader, model, optimizer, loss_fn, device):
    model.train()
    loop = tqdm(data_loader)
    for batch_idx, (image, mask) in enumerate(loop):
        image = image.to(device)
        mask = mask.float().to(device)
        model = model.to(device)

        # Forward
        mask_pred = model(image)
        loss = loss_fn(mask_pred, mask)

        # backward
        model.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

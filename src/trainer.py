from torch.backends import cudnn

from .data import *

import time
from tqdm import tqdm

from . import config
from .model_registry import get_model_adapter

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def getLog(log_path, str):
    import os
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, 'a+') as log:
        log.write('{}'.format(str))
        log.write('\n')

def _save_checkpoint(path, model, dataset_type, epoch, best_accuracy, net, optimizer):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_name": model,
            "dataset_type": dataset_type,
            "epoch": epoch,
            "best_accuracy": best_accuracy,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )

def _latest_checkpoint_path(model_savepath):
    root, extension = os.path.splitext(model_savepath)
    return f"{root}_latest{extension}"

def _prepare_eval_batch(values, device):
    if len(values) == 3:
        hsi_pca, aux, labels = values
        return {
            "hsi_pca": hsi_pca.to(device, non_blocking=True),
            "aux": aux.to(device, non_blocking=True),
            "label": labels.to(device, non_blocking=True),
        }

    hsi_pca, hsi, aux, labels = values
    return {
        "hsi_pca": hsi_pca.to(device, non_blocking=True),
        "hsi": hsi.to(device, non_blocking=True),
        "aux": aux.to(device, non_blocking=True),
        "label": labels.to(device, non_blocking=True),
    }

def _evaluate(
    model_adapter,
    model_bundle,
    test_loader,
    device,
    num_classes,
    epoch,
    total_epochs,
    show_progress,
):
    confusion = torch.zeros(
        (num_classes, num_classes), dtype=torch.long, device=device
    )
    iterator = test_loader
    if show_progress:
        iterator = tqdm(
            test_loader,
            desc=f"Eval {epoch + 1}/{total_epochs}",
            unit="batch",
            leave=False,
        )
    with torch.inference_mode():
        for values in iterator:
            batch = _prepare_eval_batch(values, device)
            logits = model_adapter.forward_eval(model_bundle, batch)
            predictions = logits.argmax(dim=1)
            labels = batch["label"].long()
            encoded = labels * num_classes + predictions
            confusion += torch.bincount(
                encoded, minlength=num_classes * num_classes
            ).reshape(num_classes, num_classes)

    counts = confusion.to(dtype=torch.float64)
    total = counts.sum()
    if total.item() == 0:
        return 0.0, 0.0, 0.0

    diagonal = counts.diagonal()
    class_totals = counts.sum(dim=1)
    oa = diagonal.sum() / total
    class_accuracy = torch.where(
        class_totals > 0,
        diagonal / class_totals,
        torch.zeros_like(class_totals),
    )
    aa = class_accuracy.mean()
    expected_agreement = (class_totals * counts.sum(dim=0)).sum() / total.pow(2)
    denominator = 1.0 - expected_agreement
    kappa = (oa - expected_agreement) / denominator if denominator.abs().item() > 0 else oa.new_zeros(())
    return oa.item(), aa.item(), kappa.item()

def train(
    epochs,
    lr,
    model,
    cuda,
    train_loader,
    test_loader,
    out_features,
    model_savepath,
    log_path,
    hsi_pca_wight,
    datasetType,
    resume_checkpoint=None,
):
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    cudnn.benchmark = True

    hsi_pca_wight_tensor = torch.from_numpy(hsi_pca_wight).to(device)

    model_adapter = get_model_adapter(model)
    model_bundle = model_adapter.build_model(config, datasetType, device)
    net = model_bundle["net"]
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_name = (config.get_value('optimizer_name') or 'adam').lower()
    weight_decay = float(config.get_value('weight_decay') or 0.0)
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            net.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    print(
        f"Optimizer: {optimizer.__class__.__name__}, "
        f"lr={lr}, weight_decay={weight_decay}"
    )
    start_epoch = 0
    max_acc = -1.0
    eval_start_epoch = max(1, int(config.get_value('eval_start_epoch') or 1))
    show_progress = bool(config.get_value('show_progress'))
    pause_file = config.get_value('pause_file') or ""
    checkpoint_snapshot_dir = config.get_value('checkpoint_snapshot_dir') or ""
    latest_checkpoint = _latest_checkpoint_path(model_savepath)
    if resume_checkpoint:
        try:
            checkpoint = torch.load(
                resume_checkpoint, map_location=device, weights_only=True
            )
        except TypeError:

            checkpoint = torch.load(resume_checkpoint, map_location=device)
        if checkpoint.get("model_name") not in (None, model):
            raise ValueError("Resume checkpoint model does not match.")
        if checkpoint.get("dataset_type") not in (None, datasetType):
            raise ValueError("Resume checkpoint dataset does not match.")
        net.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        max_acc = float(checkpoint.get("best_accuracy", -1.0))
        print(
            f"Resumed from {resume_checkpoint}: "
            f"epoch={start_epoch}, best_OA={max_acc:.4f}"
        )

        if not os.path.exists(model_savepath):
            _save_checkpoint(
                model_savepath,
                model,
                datasetType,
                start_epoch,
                max_acc,
                net,
                optimizer,
            )

    sum_time = 0
    total_epochs = start_epoch + epochs

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    current_time_log = 'start time: {}'.format(current_time)
    getLog(log_path, config.get_taskInfo())
    getLog(log_path, '-------------------Started Training-------------------')
    getLog(log_path, current_time_log)

    for local_epoch in range(epochs):
        epoch = start_epoch + local_epoch
        since = time.time()
        net.train()
        epoch_loss = 0.0
        epoch_batches = 0

        iterator = train_loader
        if show_progress:
            iterator = tqdm(
                train_loader,
                desc=f'Epoch {epoch+1}/{total_epochs}',
                unit='batch',
                leave=False,
            )
        for i, (hsi_pca, hsi, sar, tr_labels) in enumerate(iterator):
            batch = {
                "hsi_pca": hsi_pca.to(device),
                "hsi": hsi.to(device),
                "aux": sar.to(device),
                "label": tr_labels.to(device),
            }

            optimizer.zero_grad()
            outputs = model_adapter.forward_train(model_bundle, batch)
            compute_loss = getattr(model_adapter, "compute_loss", None)
            if compute_loss is None:
                loss = criterion(outputs, batch["label"])
            else:
                loss = compute_loss(outputs, batch["label"], criterion)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_batches += 1

        average_loss = epoch_loss / max(epoch_batches, 1)
        should_evaluate = epoch + 1 >= eval_start_epoch
        if should_evaluate:
            net.eval()
            acc1, aa, kappa = _evaluate(
                model_adapter,
                model_bundle,
                test_loader,
                device,
                out_features,
                epoch,
                total_epochs,
                show_progress,
            )

            if acc1 > max_acc:
                _save_checkpoint(
                    model_savepath,
                    model,
                    datasetType,
                    epoch + 1,
                    acc1,
                    net,
                    optimizer,
                )
                max_acc = acc1

        _save_checkpoint(
            latest_checkpoint,
            model,
            datasetType,
            epoch + 1,
            max_acc,
            net,
            optimizer,
        )
        if checkpoint_snapshot_dir:
            snapshot_path = os.path.join(
                checkpoint_snapshot_dir,
                f"epoch_{epoch + 1:03d}.pth",
            )
            _save_checkpoint(
                snapshot_path,
                model,
                datasetType,
                epoch + 1,
                max_acc,
                net,
                optimizer,
            )

        time_elapsed = time.time() - since
        sum_time += time_elapsed
        rest_time = (sum_time / (local_epoch + 1)) * (epochs - local_epoch - 1)
        currentTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        if should_evaluate:
            log = (
                currentTime
                + ' [Epoch: %d] [%.0fs, %.0fh %.0fm %.0fs] '
                  '[loss: %.4f] OA: %.4f AA: %.4f Kappa: %.4f'
                % (
                    epoch + 1,
                    time_elapsed,
                    (rest_time // 60) // 60,
                    (rest_time // 60) % 60,
                    rest_time % 60,
                    average_loss,
                    acc1,
                    aa,
                    kappa,
                )
            )
        else:
            log = (
                currentTime
                + ' [Epoch: %d] [%.0fs, %.0fh %.0fm %.0fs] '
                  '[loss: %.4f] evaluation skipped (starts at epoch %d)'
                % (
                    epoch + 1,
                    time_elapsed,
                    (rest_time // 60) // 60,
                    (rest_time // 60) % 60,
                    rest_time % 60,
                    average_loss,
                    eval_start_epoch,
                )
            )
        print(log)
        getLog(log_path, log)

        if pause_file and os.path.exists(pause_file):
            pause_message = (
                f"Pause requested after checkpoint at epoch {epoch + 1}."
            )
            print(pause_message)
            getLog(log_path, pause_message)
            return False

    print('max_acc: %.4f' %(max_acc))
    finish_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    finish_time_log = 'finish time: {} '.format(finish_time)
    mac_acc_log = 'max_acc: {} '.format(max_acc)
    getLog(log_path, mac_acc_log)
    getLog(log_path, finish_time_log)
    getLog(log_path, '-------------------Finished Training-------------------')
    return True

def myTrain(datasetType, model):
    """Train a model using the active experiment configuration."""
    channels = config.get_value('channels')
    windowSize = config.get_value('windowSize')
    out_features = config.get_value('out_features')
    cuda = config.get_value('cuda')
    lr = config.get_value('lr')
    epoch_nums = config.get_value('epoch_nums')
    batch_size = config.get_value('batch_size')
    eval_batch_size = config.get_value('eval_batch_size')
    num_workers = config.get_value('num_workers')
    random_seed = config.get_value('random_seed')
    model_savepath = config.get_value('model_savepath')
    log_path = config.get_value('log_path')
    resume_checkpoint = config.get_value('resume_checkpoint')

    print(f"Training: model={model}, epochs={epoch_nums}, lr={lr}, device={cuda}")
    set_random_seed(random_seed)
    train_loader, test_loader, trntst_loader, all_loader, hsi_pca_wight = getMyData(datasetType, channels, windowSize, batch_size, num_workers)
    if model == "SPCNet":

        test_loader.dataset.hsi = None
    test_loader = create_evaluation_loader(
        test_loader.dataset,
        eval_batch_size,
        num_workers,
        pin_memory=test_loader.pin_memory,
    )

    return train(
        epoch_nums,
        lr,
        model,
        cuda,
        train_loader,
        test_loader,
        out_features[datasetType],
        model_savepath[datasetType],
        log_path[datasetType],
        hsi_pca_wight,
        datasetType,
        resume_checkpoint=resume_checkpoint,
    )

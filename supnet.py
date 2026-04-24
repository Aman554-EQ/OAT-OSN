import os
import json
import torch
import torchvision
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import opts_thumos as opts
import time
import h5py
from iou_utils import *
from eval import evaluation_detection
from tensorboardX import SummaryWriter
from dataset import VideoDataSet, SuppressDataSet
from models import MYNET, SuppressNet
from loss_func import cls_loss_func, regress_loss_func, suppress_loss_func


# ─── JSON Serialization Helper ────────────────────────────────────────────────

def convert_to_serializable(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    return obj


# ─── SuppressNet Training ─────────────────────────────────────────────────────

def train_one_epoch(opt, model, train_dataset, optimizer):
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=opt['batch_size'], shuffle=True,
                                                num_workers=0, pin_memory=True, drop_last=False)
    epoch_cost = 0

    pbar = tqdm(train_loader, desc="Train", dynamic_ncols=True)
    for n_iter, (input_data, label) in enumerate(pbar):
        suppress_conf = model(input_data.cuda())

        loss = suppress_loss_func(label, suppress_conf)
        epoch_cost += loss.detach().cpu().numpy()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix({"loss": f"{epoch_cost / (n_iter + 1):.4f}"})

    return n_iter, epoch_cost


def eval_one_epoch(opt, model, test_dataset):
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=opt['batch_size'], shuffle=False,
                                               num_workers=0, pin_memory=True, drop_last=False)
    epoch_cost = 0

    pbar = tqdm(test_loader, desc="Eval", dynamic_ncols=True)
    for n_iter, (input_data, label) in enumerate(pbar):
        suppress_conf = model(input_data.cuda())

        loss = suppress_loss_func(label, suppress_conf)
        epoch_cost += loss.detach().cpu().numpy()

        pbar.set_postfix({"loss": f"{epoch_cost / (n_iter + 1):.4f}"})

    return n_iter, epoch_cost


def train(opt):
    writer = SummaryWriter()
    model = SuppressNet(opt).cuda()

    # ── Multi-GPU via DataParallel ──────────────────────────────────────────
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=opt["lr"], weight_decay=opt["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["lr_step"])

    train_dataset = SuppressDataSet(opt, subset="train")
    test_dataset  = SuppressDataSet(opt, subset=opt['inference_subset'])

    for n_epoch in range(opt['epoch']):
        n_iter, epoch_cost = train_one_epoch(opt, model, train_dataset, optimizer)

        writer.add_scalars('sup_data/cost', {'train': epoch_cost / (n_iter + 1)}, n_epoch)
        print("training loss(epoch %d): %f, lr - %f" % (
            n_epoch,
            epoch_cost / (n_iter + 1),
            optimizer.param_groups[0]["lr"]
        ))

        scheduler.step()
        model.eval()

        n_iter, eval_cost = eval_one_epoch(opt, model, test_dataset)

        writer.add_scalars('sup_data/eval', {'test': eval_cost / (n_iter + 1)}, n_epoch)
        print("testing loss(epoch %d): %f" % (n_epoch, eval_cost / (n_iter + 1)))

        # ── Handle .module for DataParallel when saving ─────────────────────
        raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        state = {'epoch': n_epoch + 1, 'state_dict': raw_model.state_dict()}
        torch.save(state, opt["checkpoint_path"] + "/checkpoint_suppress.pth.tar")
        if eval_cost < raw_model.best_loss:
            raw_model.best_loss = eval_cost
            torch.save(state, opt["checkpoint_path"] + "/ckp_best_suppress.pth.tar")

        model.train()

    writer.close()
    return


# ─── MYNET Frame Evaluation (used by make_dataset) ───────────────────────────

def eval_frame(opt, model, dataset):
    test_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=opt['batch_size'], shuffle=False,
                                               num_workers=0, pin_memory=True, drop_last=False)

    labels_cls = {}; labels_reg = {}; output_cls = {}; output_reg = {}
    for video_name in dataset.video_list:
        labels_cls[video_name] = []; labels_reg[video_name] = []
        output_cls[video_name] = []; output_reg[video_name] = []

    start_time = time.time()
    total_frames = 0
    epoch_cost = 0; epoch_cost_cls = 0; epoch_cost_reg = 0

    pbar = tqdm(test_loader, desc="eval_frame", dynamic_ncols=True)
    for n_iter, (input_data, cls_label, reg_label) in enumerate(pbar):
        act_cls, act_reg = model(input_data.cuda())

        cost_cls = cls_loss_func(cls_label, act_cls)
        epoch_cost_cls += cost_cls.detach().cpu().numpy()

        cost_reg = regress_loss_func(reg_label, act_reg)
        epoch_cost_reg += cost_reg.detach().cpu().numpy()

        cost = opt['alpha'] * cost_cls + opt['beta'] * cost_reg
        epoch_cost += cost.detach().cpu().numpy()

        act_cls = torch.softmax(act_cls, dim=-1)
        total_frames += input_data.size(0)

        for b in range(0, input_data.size(0)):
            video_name, st, ed, data_idx = dataset.inputs[n_iter * opt['batch_size'] + b]
            output_cls[video_name] += [act_cls[b, :].detach().cpu().numpy()]
            output_reg[video_name] += [act_reg[b, :].detach().cpu().numpy()]
            labels_cls[video_name] += [cls_label[b, :].numpy()]
            labels_reg[video_name] += [reg_label[b, :].numpy()]

        pbar.set_postfix({
            "loss": f"{epoch_cost / (n_iter + 1):.4f}",
            "cls":  f"{epoch_cost_cls / (n_iter + 1):.4f}",
            "reg":  f"{epoch_cost_reg / (n_iter + 1):.4f}",
        })

    end_time = time.time()
    working_time = end_time - start_time

    for video_name in dataset.video_list:
        labels_cls[video_name] = np.stack(labels_cls[video_name], axis=0)
        labels_reg[video_name] = np.stack(labels_reg[video_name], axis=0)
        output_cls[video_name] = np.stack(output_cls[video_name], axis=0)
        output_reg[video_name] = np.stack(output_reg[video_name], axis=0)

    cls_loss = epoch_cost_cls / n_iter
    reg_loss = epoch_cost_reg / n_iter
    tot_loss = epoch_cost / n_iter

    return cls_loss, reg_loss, tot_loss, output_cls, output_reg, labels_cls, labels_reg, working_time, total_frames


# ─── SuppressNet Test ─────────────────────────────────────────────────────────

def test(opt):
    model = SuppressNet(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"] + "/ckp_best_suppress.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    dataset = SuppressDataSet(opt, subset=opt['inference_subset'])

    test_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=opt['batch_size'], shuffle=False,
                                               num_workers=0, pin_memory=True, drop_last=False)

    labels = {}; output = {}
    for video_name in dataset.video_list:
        labels[video_name] = []; output[video_name] = []

    pbar = tqdm(test_loader, desc="Test", dynamic_ncols=True)
    for n_iter, (input_data, label) in enumerate(pbar):
        suppress_conf = model(input_data.cuda())

        for b in range(0, input_data.size(0)):
            video_name, idx = dataset.inputs[n_iter * opt['batch_size'] + b]
            output[video_name] += [suppress_conf[b, :].detach().cpu().numpy()]
            labels[video_name] += [label[b, :].numpy()]

    for video_name in dataset.video_list:
        labels[video_name] = np.stack(labels[video_name], axis=0)
        output[video_name] = np.stack(output[video_name], axis=0)

    outfile = h5py.File(opt['suppress_result_file'], 'w')
    for video_name in dataset.video_list:
        o = output[video_name]
        l = labels[video_name]
        dset_pred  = outfile.create_dataset(video_name + '/pred',  o.shape, maxshape=o.shape, chunks=True, dtype=np.float32)
        dset_pred[:, :]  = o[:, :]
        dset_label = outfile.create_dataset(video_name + '/label', l.shape, maxshape=l.shape, chunks=True, dtype=np.float32)
        dset_label[:, :] = l[:, :]
    outfile.close()
    print('complete')


# ─── Dataset Generation ───────────────────────────────────────────────────────

def make_dataset(opt):
    model = MYNET(opt).cuda()
    checkpoint = torch.load(opt["checkpoint_path"] + "/ckp_best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    dataset = VideoDataSet(opt, subset=opt['inference_subset'])

    _, _, _, output_cls, output_reg, labels_cls, labels_reg, _, _ = eval_frame(opt, model, dataset)

    proposal_dict = []
    outfile = h5py.File(opt['suppress_label_file'].format(opt['inference_subset']), 'w')

    num_class = opt["num_of_class"] - 1
    unit_size  = opt['segment_size']
    anchors    = opt['anchors']

    for video_name in tqdm(dataset.video_list, desc="make_dataset", dynamic_ncols=True):
        duration = dataset.video_len[video_name]

        for idx in range(0, duration):
            cls_anc = output_cls[video_name][idx]
            reg_anc = output_reg[video_name][idx]

            proposal_anc_dict = []
            for anc_idx in range(0, len(anchors)):
                cls = np.argwhere(cls_anc[anc_idx][:-1] > opt['threshold']).reshape(-1)

                if len(cls) == 0:
                    continue

                ed = idx + anchors[anc_idx] * reg_anc[anc_idx][0]
                length = anchors[anc_idx] * np.exp(reg_anc[anc_idx][1])
                st = ed - length

                for cidx in range(0, len(cls)):
                    label = cls[cidx]
                    tmp_dict = {}
                    tmp_dict["segment"] = [st, ed]
                    tmp_dict["score"]   = cls_anc[anc_idx][label]
                    tmp_dict["label"]   = label
                    tmp_dict["gentime"] = idx
                    proposal_anc_dict.append(tmp_dict)

            proposal_anc_dict = non_max_suppression(proposal_anc_dict, overlapThresh=opt['soft_nms'])
            proposal_dict += proposal_anc_dict

        nms_dict = non_max_suppression(proposal_dict, overlapThresh=opt['soft_nms'])

        input_table = np.zeros((duration, unit_size, num_class), dtype=np.float32)
        label_table = np.zeros((duration, num_class), dtype=np.float32)

        for proposal in proposal_dict:
            idx  = proposal["gentime"]
            conf = proposal["score"]
            cls  = proposal["label"]
            for i in range(0, unit_size):
                if idx + i < duration:
                    input_table[idx + i, unit_size - 1 - i, cls] = conf

        for proposal in nms_dict:
            idx = proposal["gentime"]
            cls = proposal["label"]
            label_table[idx:idx + 3, cls] = 1

        dset_input = outfile.create_dataset(video_name + '/input', input_table.shape,
                                            maxshape=input_table.shape, chunks=True, dtype=np.float32)
        dset_label = outfile.create_dataset(video_name + '/label', label_table.shape,
                                            maxshape=label_table.shape, chunks=True, dtype=np.float32)
        dset_input[:] = input_table
        dset_label[:] = label_table

        proposal_dict = []

    outfile.close()
    print('complete')
    return


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main(opt):
    if opt['mode'] == 'train':
        train(opt)
    if opt['mode'] == 'test':
        test(opt)
    if opt['mode'] == 'make':
        make_dataset(opt)
    return


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["checkpoint_path"]):
        os.makedirs(opt["checkpoint_path"])
    with open(opt["checkpoint_path"] + "/opts.json", "w") as opt_file:
        json.dump(opt, opt_file)

    if opt['seed'] >= 0:
        torch.manual_seed(opt['seed'])
        np.random.seed(opt['seed'])

    opt['anchors'] = [int(item) for item in opt['anchors'].split(',')]

    main(opt)
    while opt['wterm']:
        pass

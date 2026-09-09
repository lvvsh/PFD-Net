import time
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import utils.visualization as visual
from utils import data_loader
from tqdm import tqdm
import random
from utils.metrics import Evaluator
from network.SemiModel import SemiModel

start = time.time()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def update_ema_variables(model, ema_model, alpha): 
    # alpha 是动量更新率，0.99 意味着 Teacher 保留 99% 的历史记忆，只吸收 1% 的 Student 新知识，保证 Teacher 的稳定性
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)
    
class CombinedLoss(nn.Module):
    """ 组合 BCE 和 Dice，防止全黑 """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, target):
        # BCE Loss
        loss_bce = self.bce(pred, target)
        # Dice Loss
        probs = torch.sigmoid(pred)
        inter = (probs * target).sum(dim=(1,2))
        card = (probs + target).sum(dim=(1,2))
        loss_dice = 1.0 - (2. * inter + 1e-5) / (card + 1e-5)
        return loss_bce + loss_dice.mean()


def train1(train_loader, val_loader, Eva_train, Eva_train2, Eva_val, Eva_val2,
           data_name, save_path, net, ema_net, criterion, semicriterion, optimizer, use_ema, num_epoches):
    vis = visual.Visualization()
    vis.create_summary(data_name)
    global best_iou
    epoch_loss = 0
    net.train(True)
    ema_net.train(True)
    
    criterion_comb = CombinedLoss().cuda() 
    semicriterion_bce = nn.BCEWithLogitsLoss().cuda() 

    length = 0
    with tqdm(total=len(train_loader), desc=f'Eps {epoch}/{num_epoches}', unit='img') as pbar:
        for i, (A, B, mask, with_label, A_diff, B_sharp) in enumerate(train_loader):
            A, B = A.cuda(), B.cuda()
            A_diff, B_sharp = A_diff.cuda(), B_sharp.cuda()
            Y = mask.cuda()
            with_label = with_label.cuda()

            optimizer.zero_grad()
            
            # 初始化各类 Loss
            loss_sup = torch.tensor(0.0).cuda()
            loss_semi = torch.tensor(0.0).cuda()
            loss_null = torch.tensor(0.0).cuda() 

            # ===============================================
            # 1. 监督分支 (Supervised) - 仅处理有标签数据
            # ===============================================
            if with_label.any():
                # Student 预测四种组合 (干净 + 各种物理退化)
                p_clean = net(A[with_label], B[with_label])
                p_diffA = net(A_diff[with_label], B[with_label])
                p_diffB = net(A[with_label], B_sharp[with_label])
                p_diffAB = net(A_diff[with_label], B_sharp[with_label])
                
                y_lbl = Y[with_label]
                
                # 【导师调优】：区分主线任务和辅助数据增强任务的权重
                loss_clean = criterion_comb(p_clean[0], y_lbl) + criterion_comb(p_clean[1], y_lbl)
                loss_aug = (criterion_comb(p_diffA[0], y_lbl) + criterion_comb(p_diffA[1], y_lbl) +
                            criterion_comb(p_diffB[0], y_lbl) + criterion_comb(p_diffB[1], y_lbl) +
                            criterion_comb(p_diffAB[0], y_lbl) + criterion_comb(p_diffAB[1], y_lbl)) / 3.0
                
                loss_sup = 0.6 * loss_clean + 0.4 * loss_aug

            # ===============================================
            # 2. 半监督分支 (EMA Consistency) - 处理无标签数据
            # ===============================================
            if use_ema and (~with_label).any():
                with torch.no_grad():
                    # Teacher 生成干净原图的伪标签
                    pseudo_attn, pseudo_preds = ema_net(A[~with_label], B[~with_label])
                    conf = torch.sigmoid(pseudo_preds)
                    
                    # 【致命Bug 1 修复】：必须同时取高确信度变化(>=0.85)和高确信度不变(<=0.15)
                    mask_conf = ((conf >= 0.85) | (conf <= 0.15)).float() 
                    
                    pseudo_attn = torch.sigmoid(pseudo_attn).detach()
                    pseudo_preds = conf.detach()

                # Student 预测无标签的各种物理退化版本
                s_p_clean = net(A[~with_label], B[~with_label])
                s_p_diffA = net(A_diff[~with_label], B[~with_label])
                s_p_diffB = net(A[~with_label], B_sharp[~with_label])
                s_p_diffAB = net(A_diff[~with_label], B_sharp[~with_label])

                def calc_semi(preds, target):
                    l_attn = semicriterion_bce(preds[0], target)
                    l_pred = semicriterion_bce(preds[1], target)
                    return ((l_attn + l_pred) * mask_conf).mean()

                # 【导师调优】：一致性损失主次划分
                l_semi_clean = calc_semi(s_p_clean, pseudo_preds)
                l_semi_aug = (calc_semi(s_p_diffA, pseudo_preds) + 
                              calc_semi(s_p_diffB, pseudo_preds) +
                              calc_semi(s_p_diffAB, pseudo_preds)) / 3.0
                              
                loss_semi = 0.6 * l_semi_clean + 0.4 * l_semi_aug
                             
                # 记录指标
                Eva_train2.add_batch(Y[~with_label].cpu().numpy().astype(int), (s_p_clean[1]>0).cpu().numpy().astype(int))

            # ===============================================
            # 3. 物理退化先验分支 (Null-Change) - 全员参与
            # ===============================================
            if use_ema: 
                # A 和 A_diff 是同一图像的不同物理状态，理论上绝对无变化
                p_null = net(A, A_diff)
                y_zero = torch.zeros_like(Y)
                # 【致命Bug 2 修复】：全黑目标禁止使用 Dice Loss，直接使用稳健的 MSE
                loss_null = F.mse_loss(torch.sigmoid(p_null[0]), y_zero) + F.mse_loss(torch.sigmoid(p_null[1]), y_zero)

            # ===============================================
            # 反向传播与优化
            # ===============================================
            w_semi = 0.2 if use_ema else 0.0
            w_null = 0.1 if use_ema else 0.0
            
            total_loss = loss_sup + w_semi * loss_semi + w_null * loss_null

            if total_loss.requires_grad:
                total_loss.backward()
                # 梯度裁剪，防止 Mamba 在初期梯度爆炸 (可选，推荐保留)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                optimizer.step()

            # EMA 更新 Teacher
            if use_ema:
                with torch.no_grad():
                    update_ema_variables(net, ema_net, alpha=0.99)

            epoch_loss += total_loss.item()

            if with_label.any():
                pred_eval = torch.sigmoid(p_clean[1].detach())
                pred_eval = (pred_eval >= 0.5).int().cpu().numpy()
                target_eval = y_lbl.cpu().numpy().astype(int)
                Eva_train.add_batch(target_eval, pred_eval)

            pbar.set_postfix({'LSup': f"{loss_sup.item():.2f}", 'LSemi': f"{loss_semi.item():.2f}", 'LNull': f"{loss_null.item():.2f}"})
            pbar.update(1)
            length += 1
            

    IoU = Eva_train.Intersection_over_Union()[1]
    Pre = Eva_train.Precision()[1]
    Recall = Eva_train.Recall()[1]
    F1 = Eva_train.F1()[1]
    train_loss = epoch_loss / length

    vis.add_scalar(epoch, IoU, 'mIoU')
    vis.add_scalar(epoch, Pre, 'Precision')
    vis.add_scalar(epoch, Recall, 'Recall')
    vis.add_scalar(epoch, F1, 'F1')
    vis.add_scalar(epoch, train_loss, 'train_loss')

    print(
        '\nEpoch [%d/%d], Loss: %.4f,\n[Training]IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (
            epoch, num_epoches, \
            train_loss, \
            Eva_train2.Intersection_over_Union()[1], Eva_train2.Precision()[1], Eva_train2.Recall()[1], Eva_train2.F1()[1]))

    if use_ema is True:
        print(
            'Epoch [%d/%d],\n[Training]IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (
                epoch, num_epoches, \
                IoU, Pre, Recall, F1))
    print("Start validating!")


    net.train(False)
    net.eval()
    ema_net.train(False)
    ema_net.eval()
    for i, (A, B, mask, filename) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            A = A.cuda()
            B = B.cuda()
            Y = mask.cuda()
            
            # Student 预测
            preds = net(A,B)[1]
            output = torch.sigmoid(preds)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            pred = output.data.cpu().numpy().astype(int)
            target = Y.cpu().numpy().astype(int)
            Eva_val.add_batch(target, pred)

            # Teacher 预测 (EMA)
            preds_ema = ema_net(A, B)[1]
            output_ema = torch.sigmoid(preds_ema)
            output_ema[output_ema >= 0.5] = 1
            output_ema[output_ema < 0.5] = 0
            Eva_val2.add_batch(target, output_ema.cpu().numpy().astype(int))
            length += 1

    # Student 指标获取
    IoU_stu = Eva_val.Intersection_over_Union()
    F1_stu = Eva_val.F1()
    print('[Student Validation] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (IoU_stu[1], Eva_val.Precision()[1], Eva_val.Recall()[1], F1_stu[1]))

    # Teacher 指标获取
    IoU_tea = Eva_val2.Intersection_over_Union()
    F1_tea = Eva_val2.F1()
    print('[Ema Validation] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (IoU_tea[1], Eva_val2.Precision()[1], Eva_val2.Recall()[1], F1_tea[1]))
    
    # 【致命Bug 3 修复】：必须以 Teacher(EMA) 的精度作为保存模型的唯一标准！
    new_iou = IoU_tea[1]    
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        print('>>> [NEW BEST!] Teacher Iou :%.4f; F1 :%.4f; Best epoch : %d <<<' % (best_iou, F1_tea[1], best_epoch))
        
        # 保存 Teacher 模型 (这是你真正用于测试和发论文的模型)
        torch.save(ema_net.state_dict(), save_path + '_train1_best_teacher_iou.pth') 
        
        # 顺带保存一下 Student 和 Optimizer 状态 (用于防断点恢复)
        student_state = {
            'best_student_net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(student_state, save_path + '_train1_best_student_iou.pth')

    print('Current Best Teacher Iou :%.4f; F1 :%.4f' % (best_iou, F1_tea[1])) # 打印历史最佳
    vis.close_summary()


if __name__ == '__main__':
    seed_everything(42)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=200, help='epoch number') 
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=4, help='training batch size') 
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--train_ratio', type=float, default=0.3, help='Proportion of the labeled images') 
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--gpu_id', type=str, default='0,1', help='train use gpu')  
    parser.add_argument('--data_name', type=str, default='LEVIR', help='the test rgb images root')
    parser.add_argument('--model_name', type=str, default='PhysFreq_DiffMamba_Semi', help='model name')
    parser.add_argument('--save_path', type=str, default='./outputLEVR_30/C2F-SemiCD/LEVR-30/')  

    opt = parser.parse_args()
    print(f'labeled ration={opt.train_ratio}, Ablation现在半监督损失函数系数为:0.2!')

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif opt.gpu_id == '0,1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        print('USE GPU 0,1')

    opt.save_path = opt.save_path + opt.data_name + '/' + opt.model_name
    
    # 请确保路径匹配你们服务器上的真实路径
    if opt.data_name == 'LEVIR':
        opt.train_root = '/home/zb/LYZ/Data/levir/train/'
        opt.val_root = '/home/zb/LYZ/Data/levir/val/'
    elif opt.data_name == 'WHU':
        opt.train_root = '/home/zb/LYZ/Data/whu/train/'
        opt.val_root = '/home/zb/LYZ/Data/whu/val/'
    elif opt.data_name == 'CDD':
        opt.train_root = '/home/zb/LYZ/Data/GZ_final/train/'
        opt.val_root = '/home/zb/LYZ/Data/GZ_final/val/'
    elif opt.data_name == 's2look':
        opt.train_root = '/home/zb/LYZ/Data/S2LOOK/train/'
        opt.val_root = '/home/zb/LYZ/Data/S2LOOK/val/'

    train_loader = data_loader.get_semiloader(opt.train_root, opt.batchsize, opt.trainsize,opt.train_ratio, num_workers=8, shuffle=True, pin_memory=False)
    val_loader = data_loader.get_test_loader(opt.val_root, opt.batchsize, opt.trainsize, num_workers=6, shuffle=False, pin_memory=False)

    Eva_train = Evaluator(num_class = 2)
    Eva_train2 = Evaluator(num_class=2)
    Eva_val = Evaluator(num_class=2)
    Eva_val2 = Evaluator(num_class=2)

    model=SemiModel().cuda()
    ema_model = SemiModel().cuda()

    for param in ema_model.parameters():
        param.detach_()

    criterion = nn.BCEWithLogitsLoss().cuda()
    semicriterion = nn.BCEWithLogitsLoss().cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0025)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_name = opt.data_name
    best_iou = 0.0

    print("Start train...")

    for epoch in range(1, opt.epoch):
        for param_group in optimizer.param_groups:
            print(f"Current LR: {param_group['lr']}")

        # 前5个epoch进行纯监督Burn-in预热
        if epoch < 5: 
            use_ema = False
            print('-> [Burn-in Phase] Supervised training only.')
        else:
            use_ema = True
            print('->[Semi-Supervised Phase] EMA Teacher activated.')

        Eva_train.reset()
        Eva_train2.reset()
        Eva_val.reset()
        Eva_val2.reset()
        
        train1(train_loader, val_loader, Eva_train, Eva_train2, Eva_val, Eva_val2, data_name, save_path, model,
              ema_model, criterion, semicriterion, optimizer, use_ema, opt.epoch)

        lr_scheduler.step()

end = time.time()
print('程序训练总时间为:', end - start)
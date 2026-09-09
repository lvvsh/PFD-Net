import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import time

# ==============================================================================
# ☢️ 开启 A100 专属 TF32 推理加速引擎 (与训练保持一致，极速出图)
# ==============================================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from utils import data_loader
from utils.metrics import Evaluator
from network.SemiModel import SemiModel # C2F-SemiCD 模型

start = time.time()

def test(test_loader, Eva_test, save_path, net):
    print(">>> 🚀 正在启动全量测试与图像生成... <<<")

    net.eval() # 开启评估模式，关闭 Dropout 和 BN 更新
    
    with torch.no_grad(): # 测试阶段全局关闭梯度计算，省显存提速
        for i, (A, B, mask, filename) in enumerate(tqdm(test_loader, desc="Testing")):
            A = A.cuda()
            B = B.cuda()
            Y = mask.cuda()
            
            # 模型推理
            preds = net(A, B)
            
            # C2F 模型输出是一个 Tuple，取深层的高精度预测 preds[1]
            output = torch.sigmoid(preds[1])
            
            # 二值化 (阈值 0.5)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            
            pred = output.data.cpu().numpy().astype(int)
            target = Y.cpu().numpy().astype(int)

            # --- 保存预测的黑白 Mask 图像 (用于论文 Figure 4) ---
            for j in range(output.shape[0]):
                probs_array = (torch.squeeze(output[j])).data.cpu().numpy()
                final_mask = probs_array * 255
                final_mask = final_mask.astype(np.uint8)
                
                # 确保保存路径存在
                os.makedirs(save_path, exist_ok=True)
                final_savepath = os.path.join(save_path, filename[j] + '.png')
                
                im = Image.fromarray(final_mask)
                im.save(final_savepath)

            # 统计指标
            Eva_test.add_batch(target, pred)

    print('\n>>> ✅ 测试完成，正在计算全局评价指标... <<<')
    
    IoU = Eva_test.Intersection_over_Union()[1]
    Pre = Eva_test.Precision()[1]
    Recall = Eva_test.Recall()[1]
    F1 = Eva_test.F1()[1]
    OA = Eva_test.OA()[1]
    Kappa = Eva_test.Kappa()[1]

    # --- 顶刊风格终端打印排版 ---
    print("="*60)
    print(f"🏆 终极测试结果 (Teacher Model) 🏆")
    print("-" * 60)
    print(f"   F1-Score  : {F1 * 100:.2f}%  <-- 你的破纪录核心指标！")
    print(f"   Precision : {Pre * 100:.2f}%")
    print(f"   Recall    : {Recall * 100:.2f}%")
    print(f"   IoU       : {IoU * 100:.2f}%")
    print(f"   OA        : {OA * 100:.2f}%")
    print(f"   Kappa     : {Kappa * 100:.2f}%")
    print("="*60)
    print(f"📂 预测掩膜 (Masks) 已全部保存至: {save_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=16, help='testing batch size') # 推理时可适当调大
    parser.add_argument('--trainsize', type=int, default=256, help='testing dataset size')
    parser.add_argument('--gpu_id', type=str, default='0', help='test use gpu')
    parser.add_argument('--data_name', type=str, default='S2look', help='the test rgb images root')
    parser.add_argument('--model_name', type=str, default='PhysFreq_Diff_C2F_TF32', help='model name')
    
    # 预测图像保存路径
    parser.add_argument('--save_path', type=str, default='./test_result/C2F-SemiCD/CDD-1_0-Teacher/') 

    opt = parser.parse_args()

    # set the device for testing
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')

    if opt.data_name == 'LEVIR':
        # 你的本地测试集路径
        opt.test_root = '/home/zb/LYZ/Data/levir/test/'
    elif opt.data_name == 'WHU':
        opt.test_root = '/home/zb/LYZ/Data/whu/test/'
    elif opt.data_name == 'CDD':
        opt.test_root = '/home/zb/LYZ/Data/GZ_final/test/'
    elif opt.data_name == 'S2look':
        opt.test_root = '/home/zb/LYZ/Data/S2LOOK/test/'

    opt.save_path = os.path.join(opt.save_path, opt.data_name, opt.model_name) + '/'
    
    # 获取 DataLoader (强烈建议加上 pin_memory=True 加速数据读取)
    test_loader = data_loader.get_test_loader(opt.test_root, opt.batchsize, opt.trainsize, num_workers=4, shuffle=False, pin_memory=True)
    
    Eva_test = Evaluator(num_class=2)
    
    # 初始化网络
    model = SemiModel().cuda()

    # ==============================================================================
    # 🚨 导师修复：绝对正确的 Teacher 权重加载方式
    # ==============================================================================
    # 请将这里的路径替换为你跑出 90.44% 的那个最佳 Teacher 权重的真实绝对路径
    # 注意：论文里一定要强调这是 5% 标签的模型！
    opt.load = '/home/zb/LYZ/C2F-SemiCD-and-C2FNet-main/outputs230/C2F-SemiCD/s2-30/s2look/PhysFreq_DiffMamba_Semi_train1_best_teacher_iou.pth' 
    
    if os.path.exists(opt.load):
        print(f'>>> 正在加载 Teacher 黄金权重: {opt.load}')
        # 直接加载 Teacher 的 state_dict (没有外层字典包裹！)
        model.load_state_dict(torch.load(opt.load, map_location='cuda:0'))
    else:
        print(f'❌ 找不到权重文件！请检查路径: {opt.load}')
        exit(1)

    # 启动测试
    test(test_loader, Eva_test, opt.save_path, model)

end = time.time()
print(f'\n⏱️ 程序测试 (Test) 总耗时: {end - start:.2f} 秒')
import os
import sys
import time
from tqdm import tqdm
import numpy as np
import paddle
import visualdl

from my_dataset import MyDataset, collate_fn, from_pred
from my_model import MyLoss, MyModel
from utils import LogWriter
from configs import BATCH_SIZE, LR, WARMUP_EPOCH, TRAIN_EPOCHS, EVAL_EPOCH, MODEL_PATH
from configs import TRAIN_IMAGE_FOLDER, TRAIN_LABEL_PATH

@paddle.no_grad()
def predict_eval(model, loader):
    '''逐batch预测结果，并将结果还原到原始序列长度'''
    model.eval()
    mse = 0
    n = 0
    sys.stdout.flush()
    time.sleep(1)
    for img, label, params in tqdm(loader):
        preds = model(img)[-1]
        for reg_p, reg_t, param in zip(preds, label, params):
            n += 1
            x, y = from_pred(reg_p.numpy(), param)
            x0, y0 = from_pred(reg_t.numpy(), param)
            mse += np.sqrt(((x0-x)/2000)**2+((y0-y)/2000)**2)
    mse/=n
    return 1/(mse+0.1)

def train(model, opt, loss_fn, train_loader, valid_loader, epochs, save_path=MODEL_PATH, eval_epoch=EVAL_EPOCH):
    print('start train.')
    start_time = time.time()
    best_total = -1
    best_epoch = 0
    log = LogWriter('train.log', clear_pre_content=False)
    vdl_path = os.path.join(save_path, 'vdl_log')
    if os.path.exists(vdl_path):
        os.system('rm -rf %s'%vdl_path)
    vdl_writer = visualdl.LogWriter(vdl_path)
    iters = 0
    for epoch in range(epochs):
        model.train()
        sys.stdout.flush()
        time.sleep(1)
        # lr = 0
        for X, Yt, _ in tqdm(train_loader):
            iters += 1
            Yp = model(X)
            loss = loss_fn(Yp, Yt)
            loss_scalar = loss.numpy()[0]
            if loss_scalar>1:
                loss_scalar = 1+np.log10(loss_scalar)
            vdl_writer.add_scalar('train/loss', loss_scalar, iters)
            vdl_writer.add_scalar('train/learning_rate', opt.get_lr(), iters)
            loss.backward()
            opt.step()
            if isinstance(opt._learning_rate,
                        paddle.optimizer.lr.LRScheduler):
                opt._learning_rate.step()
            opt.clear_grad()
        f = os.path.join(save_path, 'model.pdparams')
        paddle.save(model.state_dict(), f)
        # paddle.save(opt.state_dict(), save_path+'opt.pdopt')
        log('Save to '+f)
        if (epoch+1)%eval_epoch == 0 or epoch+1==epochs:
            score = predict_eval(model, valid_loader)
            vdl_writer.add_scalar('valid/score', score, epoch)
            # train_score = predict_eval(model, train_loader)
            # vdl_writer.add_scalar('train/score', train_score, epoch)
            if score>best_total:
                f = os.path.join(save_path, 'model_best.pdparams')
                paddle.save(model.state_dict(), f)
                # paddle.save(opt.state_dict(), save_path+'opt.pdopt')
                log('Save to %s with score = %g'%(f, score))
                best_total = score
                best_epoch = epoch+1
            log('epoch_%d, %04.0fs: score=%g, best score is %g at epoch_%d' \
                %(epoch+1, time.time()-start_time, score, best_total, best_epoch))






if __name__=='__main__':
    import warnings
    warnings.filterwarnings('ignore')
    
    np.random.seed(2021)
    paddle.seed(2021)
    LogWriter('train.log', clear_pre_content=True)
    
    ids = np.arange(100)
    np.random.shuffle(ids)
    train_ids, valid_ids = ids[:80], ids[80:]
    print(len(train_ids), len(valid_ids))
    train_dataset = MyDataset(TRAIN_IMAGE_FOLDER, TRAIN_LABEL_PATH, train_ids, argument=True)
    valid_dataset = MyDataset(TRAIN_IMAGE_FOLDER, TRAIN_LABEL_PATH, valid_ids, argument=False)

    train_loader = paddle.io.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4, collate_fn=collate_fn, shuffle=True)
    valid_loader = paddle.io.DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=4, collate_fn=collate_fn, shuffle=False)

    model = MyModel()
    batch_per_epoch = len(train_dataset)//BATCH_SIZE
    iters_all = TRAIN_EPOCHS*batch_per_epoch
    lr_scheduler = paddle.optimizer.lr.PolynomialDecay(LR, power=0.9, decay_steps=batch_per_epoch*TRAIN_EPOCHS, end_lr=0)
    lr_warmup = paddle.optimizer.lr.LinearWarmup(lr_scheduler, batch_per_epoch*WARMUP_EPOCH, 0, LR)
    opt = paddle.optimizer.Momentum(learning_rate=lr_warmup, momentum=0.9, weight_decay=4e-5, parameters=model.parameters())
    loss_func = MyLoss()

    train(model, opt, loss_func, train_loader, valid_loader, WARMUP_EPOCH+TRAIN_EPOCHS)

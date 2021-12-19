
import os
import paddle
import pandas as pd
from tqdm import tqdm

from my_dataset import MyDataset, collate_fn, from_pred, post_processing
from my_model import MyModel
from configs import MODEL_PATH, TEST_IMAGE_FOLDER, RESULT_PATH

if __name__=='__main__':
    import warnings
    warnings.filterwarnings('ignore')
    test_id = [f[:-4] for f in sorted(os.listdir(TEST_IMAGE_FOLDER)) if f.endswith('.jpg')]
    test_dataset = MyDataset(TEST_IMAGE_FOLDER, argument=False)
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = MyModel()
    params = paddle.load(os.path.join(MODEL_PATH, 'model_best.pdparams'))
    # params = paddle.load(os.path.join(MODEL_PATH, 'model.pdparams'))
    model.set_state_dict(params)
    print('Load model')

    result = {
        'data':[],
        'Fovea_X':[],
        'Fovea_Y':[],
    }
    with paddle.no_grad():
        model.eval()
        test_id_iter = iter(test_id)
        test_data_iter = iter(test_dataset.datas)
        for img, label, params in tqdm(test_loader):
            preds = model(img)[-1]
            for reg_p, param in zip(preds, params):
                x, y = from_pred(reg_p.numpy(), param)
                x, y = post_processing(next(test_data_iter)[0], (x, y))
                data_id = next(test_id_iter)
                result['data'].append(data_id)
                result['Fovea_X'].append(x)
                result['Fovea_Y'].append(y)
    df = pd.DataFrame(result)
    df.to_csv(RESULT_PATH, index=False)





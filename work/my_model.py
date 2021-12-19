
import paddle
import paddleseg

from configs import FEAT_INDICES, LOSS_WEIGHTS

class MyModel(paddle.nn.Layer):
    def __init__(self):
        super(MyModel, self).__init__()
        self.feat_indices = FEAT_INDICES
        # pretrain_url = 'https://bj.bcebos.com/paddleseg/dygraph/vit_base_patch16_384.tar.gz'
        # self.backbone = paddleseg.models.backbones.ViT_base_patch16_384(pretrained=pretrain_url)
        # pretrain_url = 'https://bj.bcebos.com/paddleseg/dygraph/resnet101_vd_ssld.tar.gz'
        # self.backbone = paddleseg.models.backbones.ResNet101_vd(pretrained=pretrain_url)
        pretrain_url = 'https://bj.bcebos.com/paddleseg/dygraph/resnet50_vd_ssld_v2.tar.gz'
        self.backbone = paddleseg.models.backbones.ResNet50_vd(pretrained=pretrain_url)
        headers = [paddle.nn.Conv2D(self.backbone.feat_channels[i], 2, 1) for i in self.feat_indices]
        self.headers = paddle.nn.LayerList(headers)

    def forward(self, x):
        # res, f_shape = self.backbone(x) # ViT has shape out
        # feats = [res[i].transpose([0,2,1]).reshape(f_shape) for i in self.feat_indices]
        feats = self.backbone(x)
        feats = [feats[i] for i in self.feat_indices]
        feats = [paddle.nn.functional.adaptive_avg_pool2d(f,1) for f in feats]
        outs = [paddle.squeeze(header(f), axis=[2,3]) for f, header in zip(feats, self.headers)]
        return outs


class MyLoss(paddle.nn.Layer):
    def __init__(self, loss_w = LOSS_WEIGHTS):
        super(MyLoss, self).__init__()
        self.loss_w = loss_w

    def forward(self, preds, label):
        loss = 0
        for pred, w in zip(preds, self.loss_w):
            loss += w*paddle.nn.functional.smooth_l1_loss(pred, label)
        return loss



if __name__=='__main__':
    from configs import INPUT_IMAGE_SHAPE, BATCH_SIZE
    I = paddle.rand(shape=(BATCH_SIZE, 3, INPUT_IMAGE_SHAPE, INPUT_IMAGE_SHAPE), dtype='float32')
    label = paddle.rand(shape=(BATCH_SIZE, 2))
    model = MyModel()
    loss_func = MyLoss()
    preds = model(I)
    loss = loss_func(preds, label)
    print(loss)



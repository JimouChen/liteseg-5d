import torch
import onnx
from skimage import io
import cv2
import warnings
import onnxruntime as rt
import numpy as np

from torchvision import transforms

from LiteSeg import liteseg
from utils import get_parse

warnings.filterwarnings('ignore')
args = get_parse()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def data_process():
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize(args.img_size),
    ])
    img_path = r'D:\files\data\test_data/39-1.tiff'
    img = io.imread(img_path)
    img = np.transpose(img[:5, :, :], (1, 2, 0))
    img = cv2.resize(img, args.img_size)
    img = np.array(img / 255, np.float32)
    img = transform(img)[None].numpy()
    return img
    # label = cv2.imread('', 0)
    # data, label = np.array(img / 255, np.float32), np.array(label / 255, np.float32)
    # data, label = transform(img), transform(label)


def to_onnx():
    model = liteseg.LiteSeg(num_class=1,
                            backbone_network='shufflenet',
                            # backbone_network='mobilenet',
                            pretrain_weight=None,
                            is_train=False).to(device)
    pth_path = r'D:\py_program\git_program\liteseg-5d\checkpoint\liteseg_shuffleNet_zdm_ep400_BCE_640x640_selfResize_best/shuffleNet_LiteSeg_ep320.pth'

    checkpoints = torch.load(pth_path)
    model.load_state_dict(checkpoints['model_state_dict'])
    model.eval()

    dummy_input = torch.randn(args.batch_size // 4, args.input_channel, *args.img_size)
    dummy_input = dummy_input.to(device)
    # input_names,output_names一定要写对应，便于c++调用
    torch.onnx.export(model,
                      dummy_input,
                      "./net.onnx",
                      opset_version=11,
                      input_names=['input_name'],
                      output_names=['output_name'])
    print('export ok')

    # Checks
    onnx_model = onnx.load('./net.onnx')  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model

    print(onnx.helper.printable_graph(onnx_model.graph))


def infer_process():
    # to_onnx()
    img_data = data_process()
    print(img_data.shape)
    sess = rt.InferenceSession('./net.onnx')
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    pred_onnx = sess.run([output_name], {input_name: img_data})

    out = np.array(pred_onnx)[0][0][0]
    print(out.shape)
    print("outputs:")
    # print(out)
    out = np.where(out > 0.8, 1, 0)
    io.imsave('out.png', out)


if __name__ == '__main__':
    infer_process()

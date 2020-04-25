from torch.autograd import Variable
import torch
import numpy as np
from model.stereo_rcnn.resnet import resnet


# Load the trained model from file
model_path="/home/shalini/Models/rcnn_local/models_stereo/stereo_rcnn_12_6477.pth"
#model_path="/home/shalini/imagenet_resnet101.pth"

stereoRCNN = resnet(('__background__', 'Car'), 101, pretrained=True)
stereoRCNN.create_architecture()
checkpoint = torch.load(model_path)
stereoRCNN.load_state_dict(checkpoint['model'])

#state_dict = model.state_dict(keep_vars=False)
# Export the trained model to ONNX
#shape=[[3, 600, 1986], [3, 600, 1986], [3] , [1], [1], [1], [1], [1], [1]]

dummy_input_1 = torch.randn(1, 3, 600, 1986).cuda()
dummy_input_2 = torch.randn(1, 3, 600,1986).cuda()
dummy_input_3 = torch.randn(1, 3).cuda()
dummy_input_4 = torch.randn(1,1).cuda()
dummy_input_5 = torch.randn(1,1).cuda()
dummy_input_6 = torch.randn(1,1).cuda()
dummy_input_7 = torch.randn(1,1).cuda()
dummy_input_8 = torch.randn(1,1).cuda()
dummy_input_9 = torch.randn(1,1).cuda()

torch.onnx.export(stereoRCNN.cuda(),
        args=(dummy_input_1, dummy_input_2,dummy_input_3, dummy_input_4,dummy_input_5, dummy_input_6,dummy_input_7, dummy_input_8,dummy_input_9),
        f="stereorcnn.onnx",
        input_names=["im_left_data","im_right_data","im_info","gt_boxes_left","gt_boxes_right","gt_boxes_merge","gt_dim_orien","gt_kpts","num_boxes"],
        output_names=["rois_left","rois_right","cls_prob","bbox_pred","dim_orien_pred","kpts_prob","left_border_prob","right_border_prob","rpn_loss_cls","rpn_loss_bbox_left_right","RCNN_loss_cls","RCNN_loss_bbox","RCNN_loss_dim_orien","RCNN_loss_kpts","rois_label"])



'''from torch.autograd import Variable
import torch
import numpy as np


# Load the trained model from file
model_path="/home/shalini/Models/Stereo-RCNN-1.0/models_stereo/stereoRCNN_27.03_1_6477.tar"
#model_path="/home/shalini/imagenet_resnet101.pth"

checkpoint = torch.load(model_path)
model=checkpoint["model"]
print(model)
state_dict = model.state_dict(keep_vars=False)
# Export the trained model to ONNX
shape=[[3, 600, 1986], [3, 600, 1986], [3], [30, 5], [30, 5], [30, 5], [30, 5], [30, 6], [1]]

dummy_input = []
for i in range(len(shape)):
    input_shape = tuple([1] + shape[i])
    x = torch.randn(input_shape).cuda()

    dummy_input.append(x)

#dummy_input = Variable(torch.randn(1, 3,224,224)) # one black and white 28 x 28 picture will be the input to the model
torch.onnx.export(model, tuple(dummy_input), "resnet.onnx")



torch.onnx.export(model,               # model being run
                  tuple(dummy_input),                         # model input (or a tuple for multiple inputs)
                  "super_resolution.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                   input_names = ['im_left_data', 'im_right_data', 'im_info', 'gt_boxes_left', 'gt_boxes_right', 'gt_boxes_merge', 'gt_dim_orien', 'gt_kpts', 'num_boxes'],   # the model's input names

                  output_names = ['rois_left', 'rois_right', 'cls_prob', 'bbox_pred', 'dim_orien_pred','kpts_prob', 'left_border_prob', 'right_border_prob','rpn_loss_cls', 'rpn_loss_bbox_left_right', 'RCNN_loss_cls', 'RCNN_loss_bbox', 'RCNN_loss_dim_orien', 'RCNN_loss_kpts', 'rois_label'], # the model's output names
                 )
'''


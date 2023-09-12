# OrgTrack
直接接入模型预测接口即可，或直接使用延时标签图像进行单个类器官图像追踪

1. 修改 main.py 中网络：

         if  model_name=="MACP":  
                """  
                网络预测接口  
                """  
                model = MACPNet.MACP.MVCnet()  
                model = model.to(device)  
                model.load_state_dict(torch.load(r'MACPNet\MACP.pth', map_location='cpu'))  

2. 或者使用网络分割好的mask进行stack后替换下面代码中的“imge”:

         if show_label==True:
             def img2bin(imge):
                 newimg = []
                 for i ,tempt in enumerate(imge) :
                     # 示例输入张量
                     input_tensor = tempt
                     np_input=np.array(input_tensor)
                     #边缘检测，hysteresisMaximum=0.05,hysteresisMinimum=0.005用于分离的高低阈值参数
                     output_tensor_edge=(outtool.DetectEdges(np.expand_dims(np_input,axis=0),gaussianSigma=2.0,hysteresisMaximum=0.05,hysteresisMinimum=0.005,foregroundThreshold=0.5)).astype('int8')
                     #分离，阈值foregroundThreshold=0.5
                     output_tensor_seperate=outtool.SeparateContours(edges=output_tensor_edge,images=np.expand_dims(np_input,axis=0),foregroundThreshold=0.5,gaussianSigma=2)
                     #填孔，过滤尺寸较小的类器官minimumArea=50
                     output_tensor=outtool.Cleanup(output_tensor_seperate,minimumArea=50,fillHoles=True,removeBorders=False)
                     # 遍历每张图像
                     newimg.append(output_tensor[0])
                 newimg = np.stack(newimg, axis=0)
                 return newimg
             tailuo = np.expand_dims(img2bin(imge),axis=1)
             return tailuo,image_name
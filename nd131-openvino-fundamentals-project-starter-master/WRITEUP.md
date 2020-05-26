# Project Write-Up

This project detects humans in a video stream using different models. The models are Tensor Flow models and are used pre-and post conversion to OpenVino projects.  This projects uses the OpenVino toolkit and is written in python. 

## Explaining Custom Layers

For the process behind converting custom layers for this project, I used extensions that already exist in OpenVino.  The models I used were in the Tensorflow Object Detection Model Zoo.  The extensions are located in eployment_tools/model_optimizer/extensions/front/tf and I found extensions for the two models I used.  

The two models/ extensions I used were: 
sd_inception_v2_coco: ssd_v2_support.json
faster_rcnn_inception_v2_coco: faster_rcnn_support.json

The full commands are:

<code>python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config ssd_inception_v2_coco_2018_01_28/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json</code>

<code>python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config faster_rcnn_inception_v2_coco_2018_01_28/pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json</code>

Some of the potential reasons for handling custom layers are that they may be new models that have layers that are not understood.  


## Comparing Model Performance

Here are the command lines I used to run the model:
<code>python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m model/ssd_inception_v2_coco_2018_01_28.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://localhost:3004/fac.ffm

</br>

python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m model/faster_rcnn_inception_v2_coco.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
</code>

My method(s) to compare models before and after conversion to Intermediate Representations
were to time the process for each run where to print the time at the beginning and end of each run.  Using the OpenVino Model was faster and more accurate. 

The difference between model accuracy pre- and post-conversion is shown below - accuracy is measured by the number of people found - 6 is the correct answer.  

 Model         | Pre  (tf)         | Post  (openvino)|
| ------------- |:-------------:| -----:|
| faster_rcnn_inception_v2_coco    | 4 | 6|
| ssd_inception_v2_coco_2018_01_28      | 17   |   15  | 

The size of the model pre- and post-conversion are shown below.  There are also XML and Mapping files produced from the conversion but those in the hundres of KB range. 

| Model         | Pre  (*.pb)         | Post  (*.bin)|
| ------------- |:-------------:| -----:|
| faster_rcnn_inception_v2_coco    | 57.2 MB | 53.2 MB |
| ssd_inception_v2_coco_2018_01_28      | 102 MB      |   100.1 MB  | 


The inference time of the model pre- and post-conversion is shown below measured in minutes

Model         | Pre  (tf)         | Post  (openvino)|
| ------------- |:-------------:| -----:|
| faster_rcnn_inception_v2_coco    | 27| 25|
| ssd_inception_v2_coco_2018_01_28      | 30    |   7  | 

## Assess Model Use Cases

Some of the potential use cases of the people counter app are voting booths but with Covid19 - this could be used to maintain social distancing and only allowing one person in the frame at a time. 

Each of these use cases would be useful because in both of these cases, you want only one person and counting the total number is useful to understand the demand. 

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these effect accuracy depending on the model.  The ssd_inception model was not very accurate and thus would not be useful for this application.  



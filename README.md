# Real-Time Object Detection and Tracking with SORT Algorithm, Kalman Filter using TensorRT

​<light>This repository contains a project that implements real-time object detection and tracking using a COCO pre-trained SSD model fine-tuned on the BDD100K dataset.</light> The project leverages the SORT (Simple Online and Realtime Tracking) algorithm, Kalman Filter, and TensorRT for optimized performance.

For a detailed explanation of how to fine-tune an MMDetection model on a custom dataset, refer to my Medium blog post: "Real-Time Object Detection And Tracking With TensorRT, Kalman Filter, and SORT Algorithm: Part 1 Training the Model".

## Fine-Tuning a COCO Pre-Trained SSD Model on BDD100K Dataset

​<light>The fine-tuning process involves adapting a COCO pre-trained model to work effectively with the BDD100K dataset, which contains different classes and annotations.</light> Here's a summarized guideline to help you through the process.

### Preparing the Dataset
Before fine-tuning, it's essential to convert the BDD100K dataset into COCO format. The conversion involves the following steps:

1. Clone the BDD100K repository.
  ``` git clone https://github.com/bdd100k/bdd100k ```
2. Set up a Python environment and install required dependencies.
3. [Download BDD100K images](https://bdd-data.berkeley.edu/) and labels for object detection.
4. Use a provided script to convert BDD100K labels to COCO annotation format, specifying the appropriate dataset partition (`train` or `val`).
   ```
    mkdir bdd100k/coco_labels
    python -m bdd100k.label.to_coco -m det \
          -i bdd100k/labels/det_20/det_${SET_NAME}.json \
          -o bdd100k/coco_labels/det_${SET_NAME}_coco.json
   ```

### Fine-Tuning Process
After preparing the dataset, follow these steps to fine-tune a COCO pre-trained SSD model:

1. **Modify the Model's Head:**
   Change the `num_classes` parameter to match BDD100K's 10 classes. This modification is done in the model's configuration file, for instance, `ssd300.py`.

2. **Adjust the Class Names:**
   Define the class names specific to BDD100K and update the train, validation, and test datasets configurations accordingly.

3. **Download a Pre-Trained Model:**
   Obtain a suitable COCO pre-trained model from the mmdetection model zoo.

4. **Edit the Configuration File:**
   Customize the training parameters such as the number of epochs, learning rate, and batch size. Ensure the number of epochs accounts for both pre-training and fine-tuning stages.

5. **Set Up the Environment (Optional - Docker):**
   Using Docker is recommended for a consistent and isolated environment. Build and run a Docker container from the provided Dockerfile.

6. **Begin Training:**
   Start fine-tuning the model by executing the `train.py` script with the edited configuration file and any additional arguments required.

This summarized guide should streamline the fine-tuning process, enabling you to adapt the COCO pre-trained SSD model for use with the BDD100K dataset effectively.



## Project Structure
```
object_detection_and_tracking/
├── README.md
├── configs/
│ ├── base/
│ │ ├── datasets/
│ │ │ └── coco_detection.py
│ │ ├── models/
│ │ │ └── ssd300.py
│ │ ├── schedules/
│ │ │ └── schedule_2x.py
│ │ └── default_runtime_ssd300.py
│ ├── ssd300_coco.py
├── docker/
│ └── Dockerfile
├── model/
│ └── ssd300_best.pth
├── tests/
│ ├── test_process.py
│ ├── test_tracker.py
│ └── test_utils.py
├── LICENSE
├── convert_tensorrt.py
├── detector.py
├── inference.py
├── inference.yaml
├── process.py
├── tracker.py
└── utils.py
```

## Getting Started 
To get started with this project, follow these steps:

1. Clone this repository to your local machine:
```
git clone https://github.com/nasigang/object_detection_tracking
```

3. Build a Docker image and create a container from the Docker image (recommended):
```
docker build -t track:v1 -f docker/Dockerfile .
docker run -v $(pwd):/workspace -it --rm --ipc host track:v1
```

3. Download the trained SSD model on BDD100K dataset from the following link: [vgg16_caffe](https://download.openmmlab.com/pretrain/third_party/vgg16_caffe-292e1171.pth), or you can train on your own with `train.py` in SSD folder in this repository.

4. Download the input video "tokyo.mp4" from the following link: [tokyo.mp4](https://drive.google.com/file/d/14MHmg6zaMcg3eqfgvhjzrYSWGczjMwIN/view)
5. Convert the trained mmdetection model to TensorRT:
```
python3 convert_tensorrt.py --config configs/yolox_x_8x8_300e_coco.py --checkpoint /path/to/checkpoint_file --save_path /save/path --device 'cuda:0' --fp16 True
```
7. Modify the configuration file inference.yaml to match your hyperparameters and input paths.
8. Start inference by running the following code:
```
python3 inference.py --infer_cfg inference.yaml
```

## Algorithm
The algorithm used in this project is based on the following papers:

[Optimized Object Tracking Technique Using Kalman Filter](https://arxiv.org/abs/2103.05467)
[Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763)

The overall algorithm workflow is as follows:

Association: Utilizes the Hungarian algorithm to match detectors and trackers using an IOU matrix.
Detection: Employs a TensorRT module of the YOLOX-x model.
min_hits: The number of consecutive frames where an object appears before a tracker is created, minimizing false positives. This parameter can be adjusted in the inference.yaml file.
max_age: The number of consecutive frames that a tracked object can go undetected (e.g., the object may leave the frame). After max_age, the tracker is deleted.
skip_frame: The number of frames that pass to the tracker without detection to decrease inference time.
License
This project is licensed under the LICENSE file in the repository.

## Acknowledgments
This project was inspired by and based on the principles outlined in the aforementioned papers and utilizes the mmdetection-to-tensorrt repository for model conversion.

```

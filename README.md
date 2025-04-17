# Probing and Inducing Combinational Creativity in Vision-Language Models
![Teaser](assets/teaser.jpeg)

This is the code repository of **Probing and Inducing Combinational Creativity in Vison-Languge Models** at **CogSci 2025**.

📝 [**arXiv**](https://arxiv.org) | 🌐 [**Project Page**](https://ppyyqq.github.io/aicc) | [**Dataset**](https://drive.google.com/drive/folders/1XIvOVwP0eVX60L-STugt_vi19Q_kAEMW?usp=drive_link)
<!-- 🤗 [**Hugging Face Space**](https://huggingface.co/spaces/Yzy00518/motionReFit) -->

# Getting Started

### Prerequisites  
To run the application, you need to have the following installed:  
- Python 3.10
<!-- - Required Python packages (specified in `requirements.txt`) -->

### Installation

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/PPYYQQ/aicc-code.git
    cd aicc-code
    ```

<!-- 2. **Install Python Packages**:
    ```sh
    pip install -r requirements.txt
    ``` -->

# CreativeMashup Dataset

<!-- ![Dataset](assets/dataset.png) -->

CreativeMashup is a novel benchmark that ...

### Usage

(Coming Soon!!!)

Please download the CreativeMashup dataset from [Google Drive](https://drive.google.com/drive/folders/1XIvOVwP0eVX60L-STugt_vi19Q_kAEMW?usp=drive_link). The content inside the download link will be continuously updated to ensure you have access to the most recent data.

<!-- The file structure should be like:
```plaintext
dataset/
├── base_motion
│   ├── 000000.pkl
│   ├── 000002.pkl
│   ├── ...
│   └── 029231.pkl
├── regen
│   ├── mask_all.json
│   └── part_annotations.json
├── style_transfer
│   ├── 000009_depressed.pkl
│   ├── 000009_proud.pkl
│   ├── ...
│   └── 000642_sexy.pkl
├── adjustment
│   ├── paired_data_seed0_15_batch0_id3.pkl
│   ├── paired_data_seed0_15_batch0_id5.pkl
│   ├── ...
│   └── paired_data_seed2_13_batch4_id129.pkl
├── split
│   ├── base_motion
│   │   ├── val.txt
│   │   └── test.txt
│   ├── style_transfer
│   │   ├── val.json
│   │   └── test.json  
│   └── adjustment
│       ├── val.json
│       └── test.json  
└── README.md
```

Explanation of the files and folders of the STANCE dataset:

- **base_motion (folder):** SMPL-X format motion data from HumanML3D (24746 in total, 20 FPS).
- **regen (folder):** Motion data for body part replacement is in **base_motion**.
    - **mask_all.json:** Annotated body parts, where each key is a HumanML3D ID.
    - **part_annotations.json:** Text annotations of annotated body parts, where each key is a HumanML3D ID.
- **style_transfer (folder):** SMPL-X format motion data for motion style transfer (749 in total, 20 FPS).
- **adjustment (folder):** SMPL-X format motion data for fine-grained motion adjustment (4411 in total, 20 FPS).
- **split (folder):** Dataset split of different subsets.
    - **base_motion (folder):** Official Val & Test splits from HumanML3D.
    - **style_transfer (folder):** Val & Test pairs used in MotionReFit.
    - **adjustment (folder):** Val & Test pairs used in MotionReFit.
- Data format for `.pkl` files in **base_motion** and **style_transfer**:
    ```plaintext
    {   
        "body_pose": numpy.ndarray (N, 63),
        "global_orient": numpy.ndarray (N, 3),
        "transl": numpy.ndarray (N, 3),
    }
    ```
- Data format for `.pkl` files in **adjustment**:
    ```plaintext
    {
        "src": {   
            "body_pose": numpy.ndarray (N, 63),
            "global_orient": numpy.ndarray (N, 3),
            "transl": numpy.ndarray (N, 3),
        },
        "tgt": {   
            "body_pose": numpy.ndarray (N, 63),
            "global_orient": numpy.ndarray (N, 3),
            "transl": numpy.ndarray (N, 3),
        },
        "body_part": body part to edit,
        "text": editing instruction,
    }
    ``` -->


<!-- # Citation

```plaintext
@article{jiang2025dynamic,
  title={Dynamic Motion Blending for Versatile Motion Editing},
  author={Jiang, Nan and Li, Hongjie and Yuan, Ziye and He, Zimo and Chen, Yixin and Liu, Tengyu and Zhu, Yixin and Huang, Siyuan},
  journal={arXiv preprint arXiv:2503.20724},
  year={2025}
}
``` -->

<!-- # Related Repos

We adapted some code from other repos in data processing, training, evaluation, etc. Please check these useful repos.
```plaintext
https://github.com/jnnan/trumans_utils
https://github.com/mileret/lingo-release
https://github.com/atnikos/motionfix
https://github.com/GuyTevet/motion-diffusion-model
https://github.com/Mathux/TMR
``` -->

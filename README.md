# Restorative Step-Calibrated Diffusion

This is the code repository for our paper Step-Calibrated Diffusion for Biomedical Optical Image Restoration:

```
@misc{lyu2024stepcalibrated,
      title={Step-Calibrated Diffusion for Biomedical Optical Image Restoration}, 
      author={Yiwei Lyu and Sung Jik Cha and Cheng Jiang and Asadur Chowdury and Xinhai Hou and Edward Harake and Akhil Kondepudi and Christian Freudiger and Honglak Lee and Todd C. Hollon},
      year={2024},
      eprint={2403.13680},
      archivePrefix={arXiv}
}
```

To train the model, use the following command:

```
python3 main.py --config configs/config.yaml
```

To fine-tune the model, use the following command:

```
python3 main.py --config configs/config-finetune.yaml
```

To train the classifier, run 

```
python3 classifier.py
```

To run inference using RSCD, run the following command:

```
python3 inference/inference_recalibrate.py --config configs/config-inference-recalibrate.py
```

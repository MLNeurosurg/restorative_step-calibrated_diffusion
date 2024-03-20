# Restorative Step-Calibrated Diffusion

This is the code repository for our paper Step-Calibrated Diffusion for Biomedical Optical Image Restoration:

```
BibTex To Be Added
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
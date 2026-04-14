# MLP Tutorials on MNIST with Numpy from Scratch

```
mnist-mlp-numpy/
├── configs/
│   │── multiclass.yaml
│   │── binary.yaml
│   └── regression.yaml
├── src/
│   │── common/
│   │   ├── mnist.py                    # load_images / load_labels
│   │   ├── data.py                     # Dataset / Dataloader
│   │   ├── functions.py
│   ├   ├── modules.py
│   │   └── utils.py
│   └── train/
│       ├── wrappers.py                 # MulticlassClassifier / BinaryClassifier / Regressor
│       ├── optimizers.py               # SGD / Adam
│       └── trainers.py                 # train / evaluate / predict
├── experiments/
│   ├── 01_multiclass/
│   │   ├── 01_clf_manual.py
│   │   ├── 02_clf_module.py
│   │   ├── 03_clf_optimizer.py
│   │   ├── 04_clf_dataloader.py
│   │   ├── 05_clf_trainer.py
│   │   └── 07_clf_best.py
│   ├── 02_binary/
│   │   ├── 01_bin_manual.py
│   │   ├── 02_bin_module.py
│   │   ├── 03_bin_optimizer.py
│   │   ├── 04_bin_dataloader.py
│   │   ├── 05_bin_trainer.py
│   │   └── 07_bin_best.py
│   └── 03_regression/
│       ├── 01_reg_manual.py
│       ├── 02_reg_module.py
│       ├── 03_reg_optimizer.py
│       ├── 04_reg_dataloader.py
│       ├── 05_reg_trainer.py
│       └── 07_reg_best.py
├── notebooks/
│   ├── 01_multiclass/
│   │   ├── 01_clf_manual.ipynb         # 1.1 Manual Implementation
│   │   ├── 02_clf_module.ipynb         # 1.2 Layer Modules
│   │   ├── 03_clf_optimizer.ipynb      # 1.3 Optimizers for Training
│   │   ├── 04_clf_dataloader.ipynb     # 1.4 Custom Data Loaders
│   │   ├── 05_clf_trainer.ipynb        # 1.5 Training Wrappers
│   │   └── 07_clf_best.ipynb           # 1.6 Best Practice Configuration
│   ├── 02_binary/
│   │   ├── 01_bin_manual.ipynb         # 2.1 Manual Implementation
│   │   ├── 02_bin_module.ipynb         # 2.2 Layer Modules
│   │   ├── 03_bin_optimizer.ipynb      # 2.3 Optimizers for Training
│   │   ├── 04_bin_dataloader.ipynb     # 2.4 Custom Data Loaders
│   │   ├── 05_bin_trainer.ipynb        # 2.5 Training Wrappers
│   │   └── 07_bin_best.ipynb           # 2.6 Best Practice Configuration
│   └── 03_regression/
│       ├── 01_reg_manual.ipynb         # 3.1 Manual Implementation
│       ├── 02_reg_module.ipynb         # 3.2 Layer Modules
│       ├── 03_reg_optimizer.ipynb      # 3.3 Optimizers for Training
│       ├── 04_reg_dataloader.ipynb     # 3.4 Custom Data Loaders
│       ├── 05_reg_trainer.ipynb        # 3.5 Training Wrappers
│       └── 07_reg_best.ipynb           # 3.6 Best Practice Configuration
├── docs/
│   ├── _config.yml
│   ├── _toc.yml
│   ├── intro.md
│   └── _build/
├── .env
├── .env.example
├── requirements.txt
├── README.md
└── .gitignore

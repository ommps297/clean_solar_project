Clean Solar Panels Dust Detection - minimal working bundle
========================================================

Files included:
- train.py                : Transfer-learning training script (MobileNetV2)
- Clean_Training_Launcher.ipynb : Small notebook that explains dataset layout and runs train.py
- requirements.txt       : Python packages used
- history.json (created after first run)
- models/ (created when model is saved)
- training_plot.png (created after training)

Important notes:
- This bundle assumes you have the dataset locally in a 'data/' folder with the layout described.
- If your original repo used Git LFS, those image files were likely not downloaded; re-download the dataset images into the data/ folder.
- To run training on CPU, keep epochs small (e.g., 5-10). For real training use GPU.

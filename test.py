from ultralytics import YOLO

# 1. Load your trained 80/10/10 weights
model = YOLO("runs/detect/Ariel_Project/drone_model_final-3/weights/best.pt")

print("Evaluating model on the locked 10% TEST split...")

# 2. Force YOLO to run evaluation strictly on the test bucket
metrics = model.val(
    data="./unified_dataset_80_10_10/data.yaml",
    split="test",       # This forces it to look ONLY at the test folder
    conf=0.50,          # Keep it consistent with your deployment settings
    plots=True          # Generates the fresh confusion matrix file!
)

print("Complete! Check 'runs/detect/val/' for your final test confusion matrix.")
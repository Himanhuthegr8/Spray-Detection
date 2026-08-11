# Run the full Spray Detection pipeline: Prep -> Train -> Eval -> Infer
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  Starting Spray Detection Pipeline (Titan RTX 24GB GPU Setup)  " -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# Step 1: Prepare the YOLO Dataset (v3)
Write-Host "`n[STEP 1/4] Preparing Dataset (v3) with balanced thresholds..." -ForegroundColor Yellow
python prepare_yolo_dataset.py --output yolo_dataset_v3
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error during dataset preparation! Aborting." -ForegroundColor Red
    exit 1
}

# Step 2: Train YOLO11-seg (Resource Optimized)
Write-Host "`n[STEP 2/4] Training YOLO11-seg model..." -ForegroundColor Yellow
python train_yolo.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error during training! Aborting." -ForegroundColor Red
    exit 1
}

# Step 3: Evaluate trained model on test split
Write-Host "`n[STEP 3/4] Evaluating model performance on Test split..." -ForegroundColor Yellow
python evaluate_model.py --model runs/segment/spray_seg_titan/weights/best.pt --data yolo_dataset_v3/data.yaml --imgsz 640 --conf 0.10
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error during evaluation!" -ForegroundColor Red
    # Continue anyway to render the video
}

# Step 4: Run inference frame-by-frame on video and stitch
Write-Host "`n[STEP 4/4] Running frame-by-frame video inference (Drop.avi)..." -ForegroundColor Yellow
python infer_video.py --model runs/segment/spray_seg_titan/weights/best.pt --conf 0.10 --imgsz 640
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error during video inference!" -ForegroundColor Red
    exit 1
}

Write-Host "`n=================================================================" -ForegroundColor Green
Write-Host "  Pipeline Completed Successfully!" -ForegroundColor Green
Write-Host "  Outputs ready:" -ForegroundColor Green
Write-Host "  - Dataset: yolo_dataset_v3/" -ForegroundColor Green
Write-Host "  - Weights: runs/segment/spray_seg_titan/weights/best.pt" -ForegroundColor Green
Write-Host "  - Evaluation: evaluation_results/" -ForegroundColor Green
Write-Host "  - Video: annotated_spray.avi" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green

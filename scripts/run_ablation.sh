#!/bin/bash
# Script bash thực thi các thí nghiệm ablation study
#!/bin/bash
set -e

echo "=========================================="
echo "1/4: FULL PIPELINE (Baseline chuẩn)"
echo "=========================================="
python scripts/train_phobert.py \
    --config configs/phobert.yaml \
    --use_segment true \
    --normalize_noise true \
    --use_class_weights true \
    --save_dir checkpoints/best_phobert \
    --run_name phobert-ablation-full

echo "=========================================="
echo "2/4: BỎ TÁCH TỪ (No Word Segmentation)"
echo "=========================================="
python scripts/train_phobert.py \
    --config configs/phobert.yaml \
    --use_segment false \
    --normalize_noise true \
    --use_class_weights true \
    --save_dir checkpoints/ablation_no_segment \
    --run_name phobert-ablation-no-segment

echo "=========================================="
echo "3/4: BỎ CHUẨN HÓA TEENCODE (No Noise Norm)"
echo "=========================================="
python scripts/train_phobert.py \
    --config configs/phobert.yaml \
    --use_segment true \
    --normalize_noise false \
    --use_class_weights true \
    --save_dir checkpoints/ablation_no_norm \
    --run_name phobert-ablation-no-norm

echo "=========================================="
echo "4/4: BỎ CLASS WEIGHTS (No Class Weights)"
echo "=========================================="
python scripts/train_phobert.py \
    --config configs/phobert.yaml \
    --use_segment true \
    --normalize_noise true \
    --use_class_weights false \
    --save_dir checkpoints/ablation_no_weights \
    --run_name phobert-ablation-no-weights

echo "=========================================="
echo "TỔNG HỢP KẾT QUẢ ABLATION:"
echo "=========================================="
python -c "
import glob, json, pandas as pd
files = sorted(glob.glob('checkpoints/*/test_metrics.json'))
rows = [json.load(open(f, encoding='utf-8')) for f in files]
df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv('checkpoints/ablation_summary.csv', index=False)
print('\nĐã lưu bảng tổng hợp tại checkpoints/ablation_summary.csv')
"
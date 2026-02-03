---

## 🎯 *Mục tiêu chính của NCKH*
- Xây dựng mô hình *student nhẹ (~5–10M params)* dựa trên *CNN + Mamba*.
- Áp dụng *chiến lược KD hiệu quả* để đạt được *độ chính xác gần bằng MedNeXt*.
- Đánh giá định lượng trên *BraTS 2021/2023* (hoặc BraTS 2024 nếu có).
- So sánh với các baseline: student không KD, student + logit-KD, student + feature-KD.

---

## 📅 *Kế hoạch 4 tháng (16 tuần)*

### *Tháng 1: Chuẩn bị & Khảo sát (Tuần 1–4)*

#### Tuần 1–2: Thiết lập môi trường & dữ liệu
- [ ] Cài đặt môi trường: Python ≥3.9, PyTorch ≥2.1, MONAI, einops, causal-conv1d, mamba-ssm.
- [ ] Tải dataset *BraTS 2021 hoặc 2023* (qua [BraTS website](https://www.synapse.org/brats) hoặc qua torchio`/monai`).
- [ ] Tiền xử lý chuẩn: normalization (z-score), resampling (128×128×128 hoặc 240×240×155 tùy tài nguyên), chia train/val/test.

#### Tuần 3–4: Khảo sát & tái hiện baseline
- [ ] Tải *MedNeXt pretrained weights* (từ [MedNeXt GitHub](https://github.com/MIC-DKFZ/MedNeXt)) → chạy inference → ghi lại Dice score làm *teacher baseline*.
- [ ] Xây dựng *student baseline*:
  - Kiến trúc: U-Net nhỏ + 2–3 Mamba blocks ở encoder sâu.
  - Mục tiêu: ~7–9M params (dùng torchinfo.summary để kiểm tra).
- [ ] Huấn luyện student *không KD* → ghi lại kết quả → làm *student baseline*.

✅ **Đầu ra tháng 1**:  
- Pipeline dữ liệu hoàn chỉnh.  
- Teacher baseline (MedNeXt).  
- Student baseline (CNN+Mamba, không KD).


---

### *Tháng 2: Triển khai & thử nghiệm KD (Tuần 5–8)*

#### Tuần 5–6: Triển khai KD cơ bản
- [ ] Thêm *logit-based KD* (nhiệt độ T=2–4, alpha=0.3–0.5).
- [ ] Thêm *feature-based KD*:
  - Trích xuất features từ 3–4 stage encoder của teacher/student.
  - Dùng *L2 loss* hoặc *cosine similarity*.
  - Nếu channel mismatch → thêm adapter (1×1 conv).
- [ ] Huấn luyện student + logit-KD → student + feature-KD → so sánh.

#### Tuần 7–8: Triển khai KD nâng cao
- [ ] Thêm *boundary-aware KD*:
  - Sinh edge map từ mask teacher (dùng Sobel 3D hoặc gradient magnitude).
  - Loss: MSE giữa edge map teacher và student.
- [ ] Thử *hybrid KD*: L_total = L_task + λ1*L_feature + λ2*L_boundary.
- [ ] Tinh chỉnh hyperparameters: λ1, λ2, learning rate, scheduler.

✅ **Đầu ra tháng 2**:  
- Ít nhất 3 biến thể KD được huấn luyện và đánh giá.  
- Bảng so sánh sơ bộ: Dice, HD95, params, FLOPs.


---

### *Tháng 3: Tối ưu & Đánh giá toàn diện (Tuần 9–12)*

#### Tuần 9–10: Tối ưu mô hình & huấn luyện
- [ ] Chọn chiến lược KD tốt nhất → chạy *3 lần với seed khác nhau* → báo cáo mean ± std.
- [ ] Tối ưu data augmentation: RandAffine, RandGaussianNoise, RandFlip (theo MONAI best practice).
- [ ] Fine-tune learning rate, batch size, loss weight.

#### Tuần 11–12: Đánh giá toàn diện
- [ ] Đánh giá trên *tập test* (nếu có) hoặc *hold-out validation*.
- [ ] Metrics: *Dice (WT, TC, ET)*, *HD95*, *số tham số*, *FLOPs*.
- [ ] So sánh với:
  - MedNeXt (teacher)
  - Student w/o KD
  - Student + logit-KD
  - Student + feature-KD
  - (Tùy chọn) nnU-Net (SOTA classic)

✅ **Đầu ra tháng 3**:  
- Kết quả định lượng đầy đủ.  
- Biểu đồ so sánh (Dice bar chart, HD95 boxplot).  
- Phân tích: KD giúp cải thiện vùng nào? (ET thường khó nhất → kiểm tra riêng).


---

### *Tháng 4: Viết báo cáo & Tổng kết (Tuần 13–16)*

#### Tuần 13–14: Viết báo cáo/bài báo
- [ ] Cấu trúc đề cương:
  1. Introduction (vấn đề, động lực)
  2. Related Work (MedNeXt, Mamba, KD trong y tế)
  3. Phương pháp (kiến trúc student, chiến lược KD)
  4. Thực nghiệm (dataset, thiết lập, kết quả)
  5. Thảo luận (ưu điểm, hạn chế, hướng phát triển)
- [ ] Vẽ sơ đồ kiến trúc + pipeline KD.

#### Tuần 15–16: Kiểm tra lại & hoàn thiện
- [ ] Chạy lại experiment quan trọng để đảm bảo reproducibility.
- [ ] Đóng gói code (GitHub repo sạch, có README, requirements.txt).
- [ ] Chuẩn bị slide (nếu cần báo cáo).

✅ **Đầu ra tháng 4**:  
- Báo cáo/bài báo hoàn chỉnh.  
- Code public (nếu được phép).  
- Slide trình bày.


---

## 🧰 *Công cụ & Tài nguyên hỗ trợ*

| Mục đích | Công cụ |
|--------|--------|
| Mô hình MedNeXt | [GitHub MedNeXt](https://github.com/MIC-DKFZ/MedNeXt) |
| Mamba block | mamba_ssm, causal_conv1d (cần CUDA) |
| Data loading | MONAI, torchio |
| Đánh giá metrics | medpy, scikit-image (cho HD95) |
| Visualization | ITK-SNAP, napari, hoặc matplotlib slice-by-slice |

---

## 💡 Gợi ý tên đề tài (tiếng Việt/Anh)

- *"Efficient Knowledge Distillation from MedNeXt to CNN-Mamba Hybrid for 3D Brain Tumor Segmentation"*
- *"Nhẹ mà Mạnh: Chưng cất tri thức từ MedNeXt sang mô hình lai CNN-Mamba cho phân đoạn khối u não 3D"*

---
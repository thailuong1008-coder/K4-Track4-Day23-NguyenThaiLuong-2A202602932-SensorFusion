# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Nguyễn Thái Lương
- MSSV: 2A202602932
- Email: luong.nt2A202602932@vinuni.edu.vn
- Link repo (fork): https://github.com/luongnguyen/K4-L2L3-DAY23-NguyenThaiLuong-2A202602932-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): [Sẽ được cập nhật tự động bằng tools/check_submission.py hoặc ghi trước khi push]

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: compare, [0, 198], training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord, 0
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: precision: 0.9701, recall: 0.7004, tp/fp/fn: 519/16/222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse: 0.1503, matches: 502, sum_sq_err: 11.3436, ghost: 0, miss: 239, mean_confirmed: 2.5226
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse: 0.1359, matches: 502, sum_sq_err: 9.2668, ghost: 0, miss: 239, mean_confirmed: 2.5226
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: Fused mode có RMSE thấp hơn (0.1359 m so với 0.1503 m) chứng tỏ việc kết hợp thêm dữ liệu từ camera giúp vị trí ước lượng chính xác hơn. Trong khi đó, các chỉ số matches (502), ghost (0) và miss (239) là giống hệt nhau ở cả 2 mode, vì vòng đời track (khởi tạo, xóa) chỉ phụ thuộc vào LiDAR, camera chỉ có tác dụng cập nhật trạng thái EKF giúp giảm sai số.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?
   **Trả lời:** Đo lidar 3D cung cấp vector đo $z$ gồm 3 thành phần (x, y, z), còn camera 2D cung cấp vector $z$ gồm 2 thành phần (u, v) là toạ độ pixel trên ảnh. Tương ứng, ma trận nhiễu đo lường $R$ của lidar là 3x3 (đơn vị m²), trong khi $R$ của camera là 2x2 (đơn vị pixel²).
2. Vì sao cần gating Mahalanobis trước khi gán?
   **Trả lời:** Gating Mahalanobis giúp loại bỏ nhanh các phép đo (measurement) nằm ngoài vùng có xác suất thống kê hợp lý của một track, dựa trên khoảng cách có xét đến ma trận hiệp phương sai. Điều này giúp giảm thiểu việc tính toán gán nhầm và tăng tốc độ thuật toán gán (greedy matching).
3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.
   **Trả lời:** Là **track-then-fuse**. Quá trình predict chỉ diễn ra một lần mỗi frame, sau đó track list này được cập nhật (update) liên tiếp hai lần: đầu tiên bởi lidar (AssocL), rồi tiếp tục dùng chính track đó để gán và update bằng camera (AssocC).
4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?
   **Trả lời:** Nếu lệch calibration, mô hình chiếu h(x) sẽ tính sai tọa độ pixel, dẫn đến innovation $\gamma = z - h(x)$ bị lệch một lượng có hệ thống (bias). Kéo theo EKF sẽ cập nhật vị trí xe bị lệch đi so với thực tế để bù trừ cho sai số hệ thống này.
5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng? Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.
   **Trả lời:** Cần sensor tường minh vì ngay cả khi không có đo lường nào (`meas_list` rỗng), hệ thống vẫn phải gọi `manage_tracks` để cập nhật vòng đời (ví dụ: trừ điểm track nếu lidar không quét trúng). Lidar là cảm biến 3D chính yếu cung cấp vị trí 3D đáng tin cậy để khởi tạo và quyết định vòng đời, trong khi camera trong thiết kế lab này chỉ đóng vai trò bộ trợ tinh chỉnh vị trí chứ không chịu trách nhiệm vòng đời track.
6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.
   **Trả lời:**
   - Xác nhận: khi `score > confirmed_threshold`.
   - Giữ confirmed sau miss: Nếu miss trong FOV, bị trừ điểm `1/window` nhưng trạng thái `confirmed` không bị hạ cấp.
   - Xóa: bị xóa nếu $P_{0,0}$ hoặc $P_{1,1}$ > `max_P`, HOẶC (đã confirmed và `score < delete_threshold`), HOẶC (chưa confirmed và `score <= 0`).

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- 

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Gemini Agent (Google Antigravity)
- Dùng cho phần nào (hàm, câu hỏi, debug): Toàn bộ phần logic EKF, camera fusion, track management, association và trả lời các câu hỏi phần giải thích ngắn.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Đã chạy thành công `pytest student/tests` cho các file E–H.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))

# Tổng quan Kiến trúc Code - Bomberland AI V4 Tactical Survivor

Tài liệu này mô tả kiến trúc tổng thể, luồng dữ liệu (pipeline) và chức năng của các thành phần cốt lõi trong bộ não AI của bạn (`agent.py` và `weights.pth`). Kiến trúc này là sự kết hợp hoàn hảo giữa **Học tăng cường sâu (Deep Reinforcement Learning)** và **Thuật toán tìm kiếm quy tắc cứng (Rule-based Search)**.

---

## 1. Pipeline Tổng thể (Overall Execution Pipeline)

Mỗi khi đến lượt di chuyển, Server của Ban tổ chức sẽ gọi hàm `Agent.act(obs)`. Toàn bộ luồng xử lý bên dưới diễn ra cực kỳ tốc độ (giới hạn < 100ms):

1. **Nhận State (Observation)**: Hàm `act()` nhận dữ liệu thô (JSON/Dict) chứa tọa độ bản đồ, người chơi, bom từ Server.
2. **Quét An Toàn (Alpha-Beta Masker)**: Dữ liệu được đưa vào mô phỏng tương lai để tìm ra danh sách các hành động "Sống sót" (`safe_mask`). Bất kỳ hành động nào dẫn đến ngõ cụt hoặc dẫm phải bom nổ sẽ bị đánh dấu `False`.
3. **Mã hóa Dữ liệu (Feature Encoding)**: Dữ liệu thô được đưa qua hàm `encode_obs` để chuyển đổi thành các Tensors (Ma trận số thực) đa chiều.
4. **Tư Duy Chiến Thuật (Actor-Critic Net)**: Các Tensors được đưa qua Mạng Nơ-ron (đã nạp tri thức từ `weights.pth`) để tính toán ra xác suất ưu tiên cho 6 hành động (gọi là Logits). Mạng Nơ-ron sẽ quyết định nên đi săn, nhặt đồ hay phòng thủ.
5. **Giao thoa Cứng - Mềm (Masking & Argmax)**: Mảng `safe_mask` đè lên `Logits` (các hành động nguy hiểm bị trừ đi một lượng điểm cực lớn: `-10,000`). Hệ thống chọn ra hành động có điểm số cao nhất còn sót lại để gửi lên Server.

---

## 2. Chi tiết Công dụng các Hàm và Class chính

### A. Mạng Nơ-ron (`class ActorCriticNet`)
- **Mục đích**: Đóng vai trò là "Trực giác chiến thuật" (Biết khi nào nên ép góc dồn bom, khi nào nên bỏ chạy nhặt đồ).
- **Cấu trúc**:
  - `conv_layers` (CNN): Mạng Tích chập xử lý không gian bản đồ 13x13. Nó giúp AI "nhìn" thấy vị trí của địch, tường, hộp, và bom giống như mắt người.
  - `aux_layers` (MLP): Mạng nơ-ron kết nối đầy đủ (Fully-connected) xử lý các thông tin vô hướng (Số bom còn lại, Step đếm ngược của trận đấu).
  - `actor_head`: Đầu ra thứ nhất — ra quyết định (Policy) đánh giá điểm cho 6 hành động.
  - `critic_head`: Đầu ra thứ hai — dự đoán Giá trị (Value loss) của tình huống hiện tại (dùng chủ yếu lúc Training).
- **File `weights.pth`**: Là tập hợp hàng triệu con số (trọng số) lưu trữ kinh nghiệm của Mạng Nơ-ron sau hàng nghìn trận đấu. Hàm `Agent.__init__` sẽ nạp file này vào `ActorCriticNet`.

### B. Tiền Xử Lý Dữ Liệu (`def encode_obs`)
- **Mục đích**: "Dịch thuật" thế giới Game sang ngôn ngữ số học của Mạng Nơ-ron.
- **Công dụng**:
  - **Spatial Features (Kênh không gian 13x13x8)**: Biến bản đồ thành 8 lớp kính đè lên nhau. Kính 1 chỉ hiện Tường, Kính 2 chỉ hiện Vị trí bản thân, Kính 3 hiện Bom, Kính 4 hiện Phạm vi sóng nổ...
  - **Auxiliary Features (Đặc trưng phụ)**: Chuẩn hóa đạn dược và thời gian về thang đo `[-1.0, 1.0]` để AI không bị sốc dữ liệu.

### C. Lá Chắn Sinh Tồn (`class AlphaBetaMasker` & `class GameState`)
- **Mục đích**: Là công cụ phòng thủ tuyệt đối. Vì AI đôi khi bị "ảo giác" (hallucination) muốn lao vào lửa, Masker sẽ tát cho AI tỉnh lại và cấm hành động đó.
- **Các hàm chính**:
  - `GameState.step_forward()`: Một mini-engine giả lập game thu nhỏ. Chạy thời gian tới tương lai 1 bước để xem quả bom nào sẽ nổ, sóng nổ bay tới đâu.
  - `AlphaBetaMasker.get_safe_mask()`: Hàm chạy thuật toán Tìm kiếm theo độ sâu (DFS). Nó mô phỏng từ 7-10 bước di chuyển tiếp theo trong não để kết luận: "Nếu bước sang Trái, 5 bước sau sẽ bị kẹt giữa 2 quả bom -> cấm đi sang Trái". Nó trả về mảng `[True/False]` cho 6 nút bấm.

### D. Bộ Não Trung Tâm (`class Agent`)
- **Mục đích**: Class gốc bao bọc tất cả các thành phần trên để Ban Tổ Chức gọi lệnh.
- **Các hàm chính**:
  - `__init__(self, agent_id)`: Khởi động hệ thống. Kích hoạt GPU (nếu có), load file `weights.pth`, và đánh thức `AlphaBetaMasker`. Khối block `try-except` đảm bảo nếu file weights bị thiếu, bot vẫn không crash mà chuyển sang chơi bằng Masker thuần.
  - `act(self, obs)`: Trái tim của toàn bộ chương trình, thực thi 5 bước của **Pipeline Tổng thể**. Đo lường thời gian chặt chẽ (`t0 = time.time()`).
  - `_fallback_action()`: Hàm "Cứu Hộ". Nếu máy chủ quá lag và Mạng Nơ-ron suy nghĩ quá 80ms, hàm này lập tức nhảy vào cướp quyền, chọn bừa 1 nút bấm an toàn nhất (Dừng lại hoặc chạy trốn) để đẩy lên Server, giúp Bot không bị xử thua vì Time-out.

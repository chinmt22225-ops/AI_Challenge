# config.py
# Các tham số cấu hình cho Agent và hàm lượng giá

class Config:
    # ---------------------------------------------
    # THAM SỐ TÌM KIẾM
    # ---------------------------------------------
    TIME_LIMIT_MS = 90           # Dừng tìm kiếm khi quá giới hạn để chừa thời gian cho I/O (luật là 100ms)
    MAX_DEPTH = 6                # Độ sâu tối đa cho Iterative Deepening
    
    # ---------------------------------------------
    # TRỌNG SỐ LƯỢNG GIÁ (EVALUATION WEIGHTS)
    # ---------------------------------------------
    
    # Ưu tiên sinh tồn tuyệt đối
    SCORE_DEATH = -10000000.0    # Chết
    SCORE_SURVIVE = 10000.0      # Sống sót
    
    # Điểm phạt khi đứng vào vùng nguy hiểm (vùng bom sẽ nổ)
    SCORE_IN_DANGER = -5000.0    
    
    # Điểm thưởng nhặt vật phẩm
    SCORE_PICKUP_ITEM = 500.0
    
    # Điểm thưởng khi đặt bom phá hộp
    SCORE_BOMB_BREAK_BOX = 100.0
    
    # Điểm tiếp cận mục tiêu (ngược với khoảng cách BFS)
    # Thưởng càng nhiều khi càng gần vật phẩm
    WEIGHT_DISTANCE_ITEM = 20.0
    
    # Điểm tiếp cận kẻ địch (ngược với khoảng cách BFS)
    # Khuyến khích lại gần kẻ thù để đặt bom
    WEIGHT_DISTANCE_ENEMY = 5.0
    
    # Khoảng cách tối đa để cộng điểm (nếu xa hơn thì bỏ qua)
    MAX_DISTANCE_REWARD = 20.0
    

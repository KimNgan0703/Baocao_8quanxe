## 🧠 Các Nhóm Thuật Toán Tìm Kiếm Trong 8 Rooks Solver

---

### 🔹 **I. Tìm kiếm không có thông tin (Uninformed Search)**

#### 1️⃣ BFS (Breadth-First Search)
**BFS** mở rộng các trạng thái theo **chiều rộng**, duyệt toàn bộ các node ở cùng mức trước khi đi sâu hơn.  
Sử dụng **hàng đợi (queue)** để quản lý các trạng thái cần xét.  
Khi đặt đủ 8 quân Xe hợp lệ, thuật toán dừng lại.

#### 2️⃣ DFS (Depth-First Search)
**DFS** đi theo **chiều sâu**, mở rộng trạng thái sâu nhất trước rồi mới quay lui.  
Sử dụng **ngăn xếp (stack)** để lưu các node chưa xét.  
Khi không còn bước đi hợp lệ, thuật toán **quay lui (backtrack)**.

#### 3️⃣ UCS (Uniform Cost Search)
**UCS** mở rộng các node có chi phí nhỏ nhất.  
Mỗi lần lấy node có tổng chi phí **g(n)** thấp nhất để mở rộng, đảm bảo tìm được **lời giải tối ưu nếu chi phí đồng nhất**.

#### 4️⃣ DLS (Depth-Limited Search)
**DLS** tương tự DFS nhưng có **giới hạn độ sâu**.  
Nếu đạt đến giới hạn mà chưa có lời giải, thuật toán **cắt nhánh và quay lui** để thử hướng khác.

#### 5️⃣ IDS (Iterative Deepening Search)
**IDS** thực hiện **lặp DFS với giới hạn độ sâu tăng dần** cho đến khi tìm thấy lời giải.  
Kết hợp ưu điểm của **BFS (đầy đủ)** và **DFS (tiết kiệm bộ nhớ)**.

---

### 🔹 **II. Tìm kiếm có thông tin (Informed / Heuristic Search)**

#### 6️⃣ Greedy Best-First Search
**Greedy Best-First Search** chọn node có giá trị heuristic **h(n)** nhỏ nhất để mở rộng trước.  
Chạy nhanh nhưng **không đảm bảo tối ưu**, vì có thể rơi vào **điểm kẹt cục bộ**.

#### 7️⃣ A* (A-star Search)
Thuật toán **A\*** sử dụng công thức:
\[
f(n) = g(n) + h(n)
\]
Kết hợp giữa **chi phí thực tế (g)** và **ước lượng heuristic (h)** để tìm đường đi tối ưu.  
Nếu heuristic hợp lệ, **A\*** đảm bảo **lời giải tối ưu**.

#### 8️⃣ Beam Search
**Beam Search** chỉ giữ lại **k node tốt nhất** tại mỗi bước mở rộng.  
Giảm bộ nhớ nhưng **có thể bỏ sót lời giải tối ưu** nếu `k` quá nhỏ.

#### 9️⃣ AND-OR Tree Search
**AND-OR Tree Search** được dùng cho **môi trường không xác định**.  
- **Nút OR**: biểu diễn **lựa chọn hành động**.  
- **Nút AND**: biểu diễn **các điều kiện cần thỏa đồng thời**.  
Kết quả là một **cây kế hoạch** chứ không chỉ là đường đi đơn.

#### 🔟 Belief State Search
**Belief State Search** hoạt động trong **môi trường không chắc chắn**.  
Mỗi trạng thái là một **tập hợp khả năng (belief states)**.  
Thuật toán tìm **chuỗi hành động tối ưu** để tăng **xác suất đạt mục tiêu**.

---

### 🔹 **III. Tìm kiếm cục bộ (Local Search)**

#### 11️⃣ Hill Climbing
**Hill Climbing** xuất phát từ một trạng thái ban đầu và **leo dần lên trạng thái tốt hơn** theo heuristic.  
Thuật toán dừng khi **không còn trạng thái tốt hơn** (rơi vào **cực trị cục bộ**).

#### 12️⃣ Simulated Annealing
**Simulated Annealing** cho phép **chấp nhận tạm thời các trạng thái kém hơn** để thoát khỏi local maxima.  
**Nhiệt độ (T)** giảm dần qua mỗi vòng lặp cho đến khi đạt **ngưỡng dừng**.

#### 13️⃣ Genetic Algorithm (GA)
**Genetic Algorithm** mô phỏng **quá trình tiến hóa tự nhiên** với các thao tác:  
**chọn lọc – lai ghép – đột biến.**  
Duy trì **quần thể lời giải** và chọn **cá thể tốt nhất** sau nhiều thế hệ.

---

### 🔹 **IV. Tìm kiếm theo ràng buộc (Constraint Satisfaction Problems - CSP)**

#### 14️⃣ Backtracking
**Backtracking** là kỹ thuật **thử và sai (try & error)**.  
Đặt từng quân Xe rồi kiểm tra ràng buộc; nếu vi phạm, **quay lui** để thử hướng khác.  
Lặp lại cho đến khi tìm ra **lời giải hợp lệ**.

#### 15️⃣ Forward Checking
**Forward Checking** mở rộng từ Backtracking bằng cách **loại bỏ trước các giá trị không hợp lệ** trong miền giá trị của biến.  
Giúp **giảm đáng kể số lượng ràng buộc cần kiểm tra**, tăng tốc độ tìm kiếm.

#### 16️⃣ AC-3 (Arc Consistency 3)
**AC-3** đảm bảo **tính nhất quán cung (arc consistency)** giữa các biến trong bài toán CSP.  
Nếu một giá trị khiến ràng buộc không thỏa, nó sẽ **bị loại khỏi miền giá trị**.  
Thuật toán lặp lại cho đến khi **mọi cung đều nhất quán**.

---

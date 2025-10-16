<div align="center">
  <h1>8 Rooks Problem</h1>
  <p>Minh họa việc giải bài toán 8 quân Xe bằng các thuật toán tìm kiếm</p>
</div>

<h1 align="center">8 Rooks Problem</h1>
<h3 align="center">Minh họa việc giải bài toán 8 quân Xe bằng các thuật toán tìm kiếm</h3>

---
## Sinh viên thực hiện:

| Họ và tên | MSSV |
|------------|------|
| **Phạm Thị Kim Ngân** | **23110128** |

---

## Các Nhóm Thuật Toán Tìm Kiếm Trong 8 Rooks Solver

---
## 1. Mục tiêu
- Xây dựng chương trình giải bài toán 8 quân xe bằng các thuật toán tìm kiếm đã được học.
- Hiểu được khái niệm, cách hoạt động của từng thuật toán thông qua mô phỏng trên giao diện bàn cờ 8x8.: thòi gian chạy, số nút 
- Quan sát, đánh giá hiệu năng các thuật toán qua các tiêu chí thời gian chạy, số nút mở rộng, kích thước biên và giá trị heuristic.
## 2. Nội dung
### **2.1. Tìm kiếm không có thông tin (Uninformed Search)**
#### a. BFS (Breadth-First Search)
**BFS** mở rộng các trạng thái theo **chiều rộng**, duyệt toàn bộ các node ở cùng mức trước khi đi sâu hơn.  
Sử dụng **hàng đợi (queue)** để quản lý các trạng thái cần xét.  
Khi đặt đủ 8 quân Xe hợp lệ, thuật toán dừng lại.
- **Minh họa**:
<p align="center">
  <img src="bfs.gif" width="500" alt="BFS animation">
</p>

---

#### b. DFS (Depth-First Search)
**DFS** đi theo **chiều sâu**, mở rộng trạng thái sâu nhất trước rồi mới quay lui.  
Sử dụng **ngăn xếp (stack)** để lưu các node chưa xét.  
Khi không còn bước đi hợp lệ, thuật toán **quay lui (backtrack)**.
- **Minh họa**:
<p align="center">
  <img src="dfs.gif" width="500" alt="DFS animation">
</p>

---

#### c. UCS (Uniform Cost Search)
**UCS** mở rộng các node có chi phí nhỏ nhất.  
Mỗi lần lấy node có tổng chi phí **g(n)** thấp nhất để mở rộng, đảm bảo tìm được **lời giải tối ưu nếu chi phí đồng nhất**.
- **Minh họa**:
<p align="center">
  <img src="ucs.gif" width="500" alt="UCS animation">
</p>

---

#### d. DLS (Depth-Limited Search)
**DLS** tương tự DFS nhưng có **giới hạn độ sâu**.  
Nếu đạt đến giới hạn mà chưa có lời giải, thuật toán **cắt nhánh và quay lui** để thử hướng khác.
- **Minh họa**:
<p align="center">
  <img src="dls.gif" width="500" alt="DLS animation">
</p>

---

#### e. IDS (Iterative Deepening Search)
**IDS** thực hiện **lặp DFS với giới hạn độ sâu tăng dần** cho đến khi tìm thấy lời giải.  
Kết hợp ưu điểm của **BFS (đầy đủ)** và **DFS (tiết kiệm bộ nhớ)**.
- **Minh họa**:
<p align="center">
  <img src="ids.gif" width="500" alt="IDS animation">
</p>

---

### 2.2. Tìm kiếm có thông tin (Informed Search)
#### a. Greedy Best-First Search
**Greedy Best-First Search** chọn node có giá trị heuristic **h(n)** nhỏ nhất để mở rộng trước.  
Chạy nhanh nhưng **không đảm bảo tối ưu**, vì có thể rơi vào **điểm kẹt cục bộ**.
- **Minh họa**:
<p align="center">
  <img src="greedy.gif" width="500" alt="Greedy animation">
</p>

---

#### b. A* (A-star Search)
Thuật toán **A\*** sử dụng công thức:
\[
f(n) = g(n) + h(n)
\]
Kết hợp giữa **chi phí thực tế (g)** và **ước lượng heuristic (h)** để tìm đường đi tối ưu.  
Nếu heuristic hợp lệ, **A\*** đảm bảo **lời giải tối ưu**.
- **Minh họa**:
<p align="center">
  <img src="a_start.gif" width="500" alt="A star animation">
</p>

---

###  2.3. Tìm kiếm cục bộ (Local Search)

#### a. Hill Climbing
**Hill Climbing** xuất phát từ một trạng thái ban đầu và **leo dần lên trạng thái tốt hơn** theo heuristic.  
Thuật toán dừng khi **không còn trạng thái tốt hơn** (rơi vào **cực trị cục bộ**).
- **Minh họa**:
<p align="center">
  <img src="hill.gif" width="500" alt="Hill animation">
</p>

---

#### b. Simulated Annealing
**Simulated Annealing** cho phép **chấp nhận tạm thời các trạng thái kém hơn** để thoát khỏi local maxima.  
**Nhiệt độ (T)** giảm dần qua mỗi vòng lặp cho đến khi đạt **ngưỡng dừng**.

<p align="center">
  <img src="anna.gif" width="500" alt="Simulated Annealing animation">
</p>

---

#### c. Genetic Algorithm (GA)
**Genetic Algorithm** mô phỏng **quá trình tiến hóa tự nhiên** với các thao tác:  
**chọn lọc – lai ghép – đột biến.**  
Duy trì **quần thể lời giải** và chọn **cá thể tốt nhất** sau nhiều thế hệ.

<p align="center">
  <img src="genetic.gif" width="500" alt="Genetic Algorithm animation">
</p>

---
#### d. Beam Search
**Beam Search** chỉ giữ lại **k node tốt nhất** tại mỗi bước mở rộng.  
Giảm bộ nhớ nhưng **có thể bỏ sót lời giải tối ưu** nếu `k` quá nhỏ.
- **Minh họa**:
<p align="center">
  <img src="beam.gif" width="500" alt="Beam animation">
</p>

---

### 2.4. Tìm kiếm theo ràng buộc (Constraint Satisfaction Problems - CSP)

#### a. Backtracking
**Backtracking** là kỹ thuật **thử và sai (try & error)**.  
Đặt từng quân Xe rồi kiểm tra ràng buộc; nếu vi phạm, **quay lui** để thử hướng khác.  
Lặp lại cho đến khi tìm ra **lời giải hợp lệ**.
- **Minh họa**:
<p align="center">
  <img src="backtracking.gif" width="500" alt="Backtracking animation">
</p>

---

#### b. Forward Checking
**Forward Checking** mở rộng từ Backtracking bằng cách **loại bỏ trước các giá trị không hợp lệ** trong miền giá trị của biến.  
Giúp **giảm đáng kể số lượng ràng buộc cần kiểm tra**, tăng tốc độ tìm kiếm.
- **Minh họa**:
<p align="center">
  <img src="forward.gif" width="500" alt="Forward Checking animation">
</p>

---

#### c. AC-3 (Arc Consistency 3)
**AC-3** đảm bảo **tính nhất quán cung (arc consistency)** giữa các biến trong bài toán CSP.  
Nếu một giá trị khiến ràng buộc không thỏa, nó sẽ **bị loại khỏi miền giá trị**.  
Thuật toán lặp lại cho đến khi **mọi cung đều nhất quán**.
- **Minh họa**:
<p align="center">
  <img src="ac3.gif" width="500" alt="AC3 animation">
</p>

---
### 2.5. Nhóm thuật toán trong môi trường phức tạp (Complex Search)
#### a. AND-OR Tree Search
**AND-OR Tree Search** được dùng cho **môi trường không xác định**.  
- **Nút OR**: biểu diễn **lựa chọn hành động**.  
- **Nút AND**: biểu diễn **các điều kiện cần thỏa đồng thời**.  
Kết quả là một **cây kế hoạch** chứ không chỉ là đường đi đơn.
- **Minh họa**:
<p align="center">
  <img src="and_or.gif" width="500" alt="And-Or animation">
</p>

---

#### b. Belief State Search
**Belief State Search** hoạt động trong **môi trường không chắc chắn**.  
Mỗi trạng thái là một **tập hợp khả năng (belief states)**.  
Thuật toán tìm **chuỗi hành động tối ưu** để tăng **xác suất đạt mục tiêu**.
- **Minh họa**:
<p align="center">
  <img src="belief.gif" width="500" alt="Belief animation">
</p>
---

### 4. Cài đặt & Chạy chương trình
- pip install pillow
- pip install matplotlib
---

## Tài liệu tham khảo:
- Russell & Norvig (2016). *Artificial Intelligence: A Modern Approach (3rd Edition)*.  

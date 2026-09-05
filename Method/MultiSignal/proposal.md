# Adaptive Reward Decoding for Large Language Models

## 1. Ý tưởng chính

Khi một mô hình ngôn ngữ tạo câu trả lời, nó thường chọn token tiếp theo dựa trên xác suất mà mô hình đã học được. Cách chọn này giúp câu trả lời tự nhiên, nhưng chưa chắc đã đúng, hữu ích, an toàn hoặc phù hợp với yêu cầu của người dùng.

Dự án này muốn bổ sung các **tín hiệu hướng dẫn** vào quá trình tạo câu trả lời. Có thể hình dung mỗi tín hiệu như một người cố vấn:

- Có tín hiệu nhìn vào câu trả lời hiện tại và cho biết nó đang tốt hay xấu.
- Có tín hiệu dự đoán hướng viết hiện tại có dẫn đến một kết quả tốt hay không.
- Có tín hiệu đánh giá chi tiết từng token tiếp theo.
- Có tín hiệu phù hợp với bài toán suy luận, trong khi tín hiệu khác phù hợp với hội thoại hoặc sở thích của người dùng.

Thay vì luôn dùng một tín hiệu cố định, dự án sẽ xây dựng một **controller nhẹ** để quyết định nên tin tín hiệu nào nhiều hơn đối với từng prompt và từng giai đoạn tạo câu trả lời.

Nói ngắn gọn:

> Dự án giúp LLM tự chọn cách dùng reward phù hợp nhất với câu hỏi hiện tại, thay vì áp dụng một cách decoding giống nhau cho mọi câu hỏi.

---

## 2. Vì sao cần nhiều loại reward signal?

Không có một cách đánh giá nào luôn tốt trong mọi trường hợp.

Ví dụ:

- Với câu hỏi toán, tín hiệu kiểm tra từng bước suy luận có thể quan trọng nhất.
- Với hội thoại thông thường, tín hiệu về độ hữu ích và tự nhiên có thể phù hợp hơn.
- Với một câu trả lời dài, đánh giá phần văn bản đã sinh có thể chưa đủ; cần dự đoán xem hướng viết hiện tại sẽ dẫn đến kết quả nào.
- Có lúc reward model không chắc chắn hoặc cho tín hiệu sai. Khi đó, giữ nguyên phân phối của mô hình gốc có thể an toàn hơn.

Vì vậy, hệ thống không nên phụ thuộc hoàn toàn vào một reward model. Nó cần biết khi nào nên dùng, dùng bao nhiêu và khi nào nên bỏ qua reward guidance.

---

## 3. Các paper hiện có đóng vai trò gì?

### ARGS - Alignment as Reward-Guided Search

ARGS chấm điểm một số token có khả năng được chọn tiếp theo. Điểm cuối cùng được tạo từ hai phần:

- Mức độ tự nhiên theo mô hình gốc.
- Mức độ phù hợp theo reward model.

Điểm mạnh của ARGS là đơn giản và dễ triển khai. Tuy nhiên, reward model của nó vốn được thiết kế để chấm một câu trả lời hoàn chỉnh, nên việc dùng nó để chấm một đoạn chưa viết xong có thể chưa chính xác.

### RAD - Reward-Augmented Decoding

RAD cũng dùng reward để điều chỉnh việc chọn token, nhưng reward model được huấn luyện để đánh giá các đoạn văn bản đang được tạo dở.

Điều này giúp RAD hiểu trạng thái hiện tại tốt hơn và có thể tái sử dụng kết quả tính toán từ các token trước để giảm chi phí.

### CD - Controlled Decoding

CD không chỉ hỏi: “Đoạn hiện tại có tốt không?”, mà cố gắng dự đoán:

> Nếu tiếp tục viết từ đây, khả năng nhận được kết quả tốt trong tương lai là bao nhiêu?

Do đó, CD có góc nhìn dài hạn hơn ARGS và RAD. Prefix scorer của CD có thể được xem như một mô hình dự đoán giá trị của hướng sinh hiện tại.

### GenARM - Autoregressive Reward Model

GenARM biến reward thành tín hiệu chi tiết ở mức token. Reward model tạo ra một phân phối cho token tiếp theo, sau đó kết hợp phân phối này với mô hình gốc.

Điểm mạnh là reward phù hợp tự nhiên với cách LLM sinh văn bản từng token một. GenARM cũng hỗ trợ kết hợp nhiều mục tiêu, nhưng thường phải dùng các trọng số được chọn trước.

### TARo - Token-level Adaptive Routing

TARo dùng một mạng MLP nhỏ để điều chỉnh mức ảnh hưởng của mô hình gốc và reward model theo từng token.

Đây là công trình gần nhất với ý tưởng controller của dự án. Vì vậy, TARo nên được xem là **baseline quan trọng và nền tảng kiến trúc**, không phải một loại reward signal mới.

Điểm dự án cần phát triển xa hơn TARo là:

- TARo chủ yếu điều chỉnh giữa mô hình gốc và một reward model.
- Dự án muốn lựa chọn giữa nhiều cách nhìn reward khác nhau.
- Dự án còn quan tâm đến độ tin cậy và chi phí của từng tín hiệu.

---

## 4. Hệ thống dự kiến hoạt động như thế nào?

### Bước 1: Mô hình gốc đề xuất token

LLM tạo ra danh sách các token có khả năng xuất hiện tiếp theo.

### Bước 2: Các reward expert đưa ra ý kiến

Mỗi reward expert đánh giá các token theo một góc nhìn riêng, chẳng hạn:

- Điểm reward của đoạn hiện tại.
- Mức thay đổi reward nếu thêm token mới.
- Giá trị dài hạn của hướng sinh.
- Reward trực tiếp ở mức token.
- Tín hiệu về suy luận, tính đúng đắn hoặc sở thích người dùng.

Các paper mới tìm được sau này có thể được thêm vào hệ thống nếu chúng tạo ra một tín hiệu hữu ích và có thể chuyển thành điểm cho các lựa chọn decoding.

### Bước 3: Chuẩn hóa các điểm

Mỗi reward expert có cách tính và khoảng giá trị khác nhau. Không thể cộng trực tiếp các điểm này vì expert có số lớn hơn sẽ dễ lấn át các expert còn lại.

Do đó, hệ thống phải đưa chúng về một thang đo tương đối giống nhau trước khi kết hợp.

### Bước 4: Controller quyết định mức độ tin tưởng

Controller nhận các thông tin như:

- Nội dung và loại prompt.
- Mô hình gốc đang chắc chắn hay phân vân.
- Các reward expert có đồng ý với nhau không.
- Expert nào đang không chắc chắn.
- Câu trả lời đã sinh đến giai đoạn nào.
- Chi phí tính toán đã sử dụng.

Sau đó controller tạo ra trọng số cho từng expert.

### Bước 5: Chọn token và tiếp tục

Điểm của mô hình gốc và các reward expert được kết hợp để chọn token tiếp theo. Quá trình này lặp lại cho đến khi hoàn thành câu trả lời.

Hệ thống luôn cần một lựa chọn **base-only**. Khi reward không đáng tin, controller có thể để mô hình gốc tự decoding.

---

## 5. Controller hai tầng

Nếu chạy tất cả reward model ở mọi token, hệ thống sẽ rất chậm. Vì vậy, có thể chia controller thành hai tầng.

### Prompt-level controller

Đọc prompt và chọn trước một hoặc hai reward expert có khả năng phù hợp nhất.

Ví dụ:

- Prompt toán ưu tiên expert về suy luận.
- Prompt hội thoại ưu tiên expert về helpfulness.
- Prompt cần viết sáng tạo có thể giảm mức can thiệp của reward.

### Token-level controller

Trong quá trình sinh, controller tiếp tục thay đổi mức ảnh hưởng của những expert đã chọn.

Ví dụ, ở phần suy luận nó có thể tin reward expert nhiều hơn, nhưng ở phần trình bày câu trả lời cuối cùng lại ưu tiên độ tự nhiên của mô hình gốc.

Thiết kế này vừa linh hoạt vừa tránh phải chạy mọi expert liên tục.

---

## 6. Điểm mới mà dự án cần hướng tới

Chỉ sử dụng MLP để chọn phương pháp là chưa đủ mạnh, vì TARo đã có router ở mức token. Điểm mới của dự án nên nằm ở các nội dung sau:

1. **Thống nhất nhiều cách nhìn reward**
   
   Đưa reward toàn câu, reward theo prefix, dự đoán giá trị tương lai và reward theo token vào cùng một framework.

2. **Lựa chọn reward theo tình huống**
   
   Controller không chỉ dùng prompt mà còn theo dõi trạng thái trong quá trình sinh.

3. **Biết khi nào không nên tin reward**
   
   Hệ thống sử dụng độ bất định và mức bất đồng giữa các expert để tránh guidance sai.

4. **Cân bằng chất lượng và chi phí**
   
   Controller ưu tiên expert tốt nhưng cũng phải tính đến latency, bộ nhớ và số lần chạy model.

5. **Hoạt động trên domain và model mới**
   
   Controller cần được kiểm tra trên những loại prompt và backbone chưa xuất hiện khi huấn luyện.

---

## 7. Câu hỏi nghiên cứu chính

Dự án có thể xoay quanh các câu hỏi sau:

1. Các loại reward signal có thật sự phù hợp với những nhóm prompt khác nhau không?
2. Có thể tự động lựa chọn reward signal tốt hơn việc dùng một phương pháp cố định không?
3. Lựa chọn theo từng token có tốt hơn chỉ lựa chọn một lần theo prompt không?
4. Độ bất định và mức bất đồng giữa các expert có giúp tránh reward guidance sai không?
5. Controller có chuyển sang domain, model family và model scale khác mà không cần huấn luyện lại không?
6. Chất lượng tăng thêm có xứng đáng với chi phí inference hay không?

---

## 8. Thí nghiệm quan trọng nhất: Oracle Study

Trước khi xây controller phức tạp, cần chạy tất cả phương pháp trên cùng một tập prompt và kiểm tra phương pháp nào tốt nhất cho từng prompt.

Mục tiêu của bước này là trả lời:

- Có phải một phương pháp luôn thắng không?
- Hay mỗi phương pháp thắng trên một nhóm prompt khác nhau?
- Nếu luôn được chọn đúng phương pháp, kết quả có cao hơn đáng kể so với phương pháp cố định tốt nhất không?

Kết quả “luôn chọn đúng phương pháp” được gọi là **oracle result**.

Nếu oracle chỉ tốt hơn rất ít, việc xây controller có thể không mang lại nhiều ý nghĩa. Nếu oracle tốt hơn rõ rệt, đây sẽ là bằng chứng mạnh cho thấy bài toán routing thật sự cần thiết.

---

## 9. Cách đánh giá

Không nên chỉ dùng chính reward model đã hướng dẫn decoding để chấm kết quả. Điều đó có thể khiến hệ thống đạt điểm reward cao nhưng chất lượng thực tế không tăng.

Cần đánh giá bằng nhiều nguồn độc lập:

- Độ chính xác trên bài toán có đáp án rõ ràng.
- Evaluator hoặc judge không tham gia vào quá trình decoding.
- So sánh cặp câu trả lời trên tập preference test riêng.
- Độ tự nhiên, mạch lạc và đa dạng.
- Mức thay đổi so với mô hình gốc.
- Latency, tokens mỗi giây, FLOPs và bộ nhớ.
- Kết quả trên domain chưa thấy.

Các thí nghiệm chính cần so sánh:

- Base model không dùng reward.
- Từng phương pháp ARGS, RAD, CD và GenARM riêng lẻ.
- Kết hợp đều tất cả expert.
- Chọn expert ngẫu nhiên.
- Chọn một expert theo prompt.
- Controller theo từng token.
- TARo.
- Oracle chọn expert tốt nhất.

---

## 10. Hướng tìm thêm paper

Có thể tiếp tục tìm các paper tạo reward signal thuộc những nhóm sau:

- Process reward model cho reasoning.
- Outcome reward model cho câu trả lời hoàn chỉnh.
- Value model dự đoán kết quả tương lai từ prefix.
- Token-level hoặc step-level reward.
- Reward model có uncertainty hoặc calibration.
- Multi-objective reward model.
- Verifier cho toán, code và factuality.
- Self-evaluation hoặc critique model.
- Reward model nhỏ có thể hướng dẫn model lớn.
- Reward signal có chi phí thấp hoặc hỗ trợ early exit.

Khi tìm được paper mới, cần kiểm tra bốn câu hỏi:

1. Paper tạo ra tín hiệu gì?
2. Tín hiệu được tính ở mức toàn câu, từng bước, prefix hay token?
3. Có thể dùng tín hiệu đó trong lúc decoding không?
4. Nó bổ sung điều gì mà các expert hiện tại chưa có?

Không nên thêm paper chỉ để tăng số lượng expert. Một expert mới chỉ có ích nếu nó cung cấp góc nhìn khác hoặc hoạt động tốt trên một nhóm prompt riêng.

---

## 11. Lộ trình trước mắt

### Giai đoạn 1: Xây nền tảng chung

- Chọn một base model nhỏ để thử nghiệm.
- Chuẩn hóa cách chạy và đánh giá ARGS, RAD, CD và GenARM.
- Dùng cùng prompt, cùng cách sampling và cùng evaluator.

### Giai đoạn 2: Kiểm tra giả thuyết

- Chạy oracle study.
- Phân tích loại prompt mà mỗi expert làm tốt.
- Đo mức bất đồng giữa các expert.
- Kiểm tra fixed mixture có tốt hơn từng expert riêng lẻ không.

### Giai đoạn 3: Xây controller

- Bắt đầu bằng prompt-level MLP đơn giản.
- So sánh với random routing và rule-based routing.
- Sau đó mới thêm token-level controller.
- Thêm lựa chọn base-only và cost penalty.

### Giai đoạn 4: Mở rộng cho bài tạp chí

- Thử ít nhất hai model family và nhiều model size.
- Đánh giá trên nhiều loại nhiệm vụ.
- Chạy thí nghiệm chuyển domain và weak-to-strong.
- Phân tích latency, bộ nhớ và chất lượng.
- Bổ sung phần lý giải thống nhất các reward expert.

---

## 12. Rủi ro cần chú ý

- Các reward expert có thể quá giống nhau nên controller không học được điều có ích.
- Expert có thang điểm lớn hơn có thể lấn át expert khác nếu chưa chuẩn hóa.
- Chạy nhiều reward model có thể làm inference quá chậm.
- Controller có thể chỉ học nhận diện dataset thay vì học cách chọn reward.
- Dùng cùng một reward model để hướng dẫn và đánh giá có thể tạo kết quả thiếu khách quan.
- Một reward expert kém có thể làm giảm chất lượng nếu controller không có lựa chọn bỏ qua.
- Nếu chỉ thử trên một model và một domain, chưa đủ chứng minh khả năng tổng quát hóa.

---

## 13. Đóng góp kỳ vọng của bài báo

Nếu dự án được triển khai đầy đủ, bài báo có thể đưa ra các đóng góp sau:

1. Một cách nhìn thống nhất đối với nhiều loại reward-guided decoding.
2. Một controller nhẹ lựa chọn và phối hợp reward signal theo prompt và từng bước sinh.
3. Cơ chế sử dụng độ bất định và chi phí để tránh guidance không phù hợp.
4. Kết quả thực nghiệm cho thấy không có reward expert nào tốt nhất trong mọi trường hợp.
5. Khả năng chuyển controller sang domain và backbone khác.
6. Phân tích rõ sự đánh đổi giữa chất lượng và chi phí inference.

---

## 14. Tên đề tài tạm thời

### Tên tiếng Anh

**MoRE-Decoding: Budget-Aware Routing over Multi-Granularity Reward Experts for Adaptive LLM Decoding**

### Tên tiếng Việt

**MoRE-Decoding: Định tuyến thích nghi các tín hiệu reward đa mức cho quá trình giải mã của mô hình ngôn ngữ lớn**

Tên này chỉ là tên tạm thời và có thể thay đổi sau khi hoàn thành oracle study và xác định đóng góp mạnh nhất của dự án.


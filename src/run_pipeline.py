"""Run the full pipeline end to end:  uv run python -m src.run_pipeline"""
import time #time là thư viện có sẵn của Python. Trong file này, nó được dùng để đo thời gian thực hiện từng bước.

from src.data import clean_data
from src.fairness import fairness_policies
from src.features import build_features
from src.models import evaluate, leakage_audit, score_establishments, train_models
from src.report import build_memo
#5 dòng trên import các module con từ các thư mục khác nhau trong dự án. Mỗi module con này chứa các hàm và logic cần thiết để thực hiện các bước khác nhau trong pipeline.
STEPS = [
    ("1. Clean raw data", clean_data.main),
    ("2. Point-in-time features", build_features.main),
    ("3. Train + calibrate models", train_models.main),
    ("4. Evaluate on test year", evaluate.main),
    ("5. Leakage audit", leakage_audit.main),
    ("6. Fairness + targeting policies", fairness_policies.main),
    ("7. Score establishments (priority list)", score_establishments.main),
    ("8. Render policy memo to Word", build_memo.main),
]
#phần steps là một danh sách các bước trong pipeline, mỗi bước được biểu diễn bằng một tuple gồm tiêu đề và file thực hiện bước đó.
if __name__ == "__main__": #dòng này kiểm tra xem file này có được chạy trực tiếp hay không. Nếu đúng, nó sẽ thực hiện các bước trong pipeline.
    for title, step in STEPS: #duyệt từng bước trong danh sách STEPS. Mỗi bước bao gồm tiêu đề và hàm thực hiện.
        t0 = time.time() #lưu thời gian bắt đầu thực hiện bước hiện tại để tính toán thời gian thực hiện.
        print(f"\n===== {title} =====")
        step() #gọi hàm thực hiện bước hiện tại. Mỗi hàm này sẽ thực hiện các thao tác cần thiết để hoàn thành bước đó trong pipeline.
        print(f"----- done in {time.time() - t0:.0f}s") #in ra thời gian thực hiện bước hiện tại bằng cách tính toán sự khác biệt giữa thời gian hiện tại và thời gian bắt đầu đã lưu trước đó.

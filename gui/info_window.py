from tkinter import Toplevel, Label, Button

def display_info(language, root):
    info_window = Toplevel(root)
    info_window.title("Info")
    info_window.geometry("700x600+100+100")

    info_text = {
        'English': [
            "age - age in years",
            "sex - sex (1 = male; 0 = female)",
            "cp - chest pain type(0 = typical angina; 1 = atypical angina; 2 = non-anginal pain; 3 = asymptomatic)",
            "trestbps - resting blood pressure (in mm Hg on admission to the hospital)",
            "chol - serum cholesterol in mg/dl",
            "fbs - fasting blood sugar > 120 mg/dl (1 = true; 0 = false)",
            "restecg - resting electrocardiographic results (0 = normal; 1 = ST-T wave abnormality; 2 = showing probable or definite left ventricular hypertrophy)",
            "thalach - maximum heart rate achieved",
            "exang - exercise-induced angina (1 = yes; 0 = no)",
            "oldpeak - ST depression induced by exercise relative to rest",
            "slope - the slope of the peak exercise ST segment (0 = upsloping; 1 = flat; 2 = downsloping)",
            "ca - number of major vessels (0-3) colored by fluoroscopy",
            "thal - Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)"
        ],
        'Vietnamese': [
            "age - tuổi tính bằng năm",
            "sex - giới tính (1 = nam; 0 = nữ)",
            "cp - loại đau ngực (0 = đau thắt ngực điển hình; 1 = đau thắt ngực không điển hình; 2 = đau không do mạch vành; 3 = không có triệu chứng)",
            "trestbps - huyết áp nghỉ ngơi (mm Hg khi nhập viện)",
            "chol - cholesterol huyết thanh (mg/dl)",
            "fbs - đường huyết đói > 120 mg/dl (1 = đúng; 0 = sai)",
            "restecg - kết quả điện tâm đồ nghỉ (0 = bình thường; 1 = ST-T bất thường; 2 = phì đại thất trái rõ rệt hoặc chắc chắn)",
            "thalach - nhịp tim tối đa đạt được",
            "exang - đau thắt ngực do tập thể dục (1 = có; 0 = không)",
            "oldpeak - mức độ trầm cảm ST do tập thể dục so với lúc nghỉ",
            "slope - độ dốc của đoạn ST khi tập thể dục (0 = lên dốc; 1 = phẳng; 2 = xuống dốc)",
            "ca - số lượng mạch chính (0-3) được làm nổi bật bởi fluoroscopy",
            "thal - Thalassemia (0 = bình thường; 1 = khiếm khuyết cố định; 2 = khiếm khuyết hồi phục)"
           
        ]
    }

    Label(info_window, text="Information related to dataset", font="robot 15 bold").pack(padx=20, pady=20)
    for idx, text in enumerate(info_text[language]):
        Label(info_window, text=text, font="arial 12").place(x=20, y=100 + 30 * idx)

def Info():
    info_window = Toplevel()
    info_window.title("Info")
    info_window.geometry("400x200+100+100")

    Label(info_window, text="Choose Language", font="robot 15 bold").pack(padx=20, pady=20)

    Button(info_window, text="English", font="arial 12", command=lambda: display_info('English', info_window)).pack(pady=10)
    Button(info_window, text="Vietnamese", font="arial 12", command=lambda: display_info('Vietnamese', info_window)).pack(pady=10)
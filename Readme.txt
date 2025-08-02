pip install -r requirements.txt // cài thư viện

deloy web:

terminal
+ chạy be: 
	cd backend
	uvicorn main:app --reload --host 0.0.0.0 --port 8000
+ chạy fe:
	cd frontend
	python -m http.server 8001

bảng thủ ngữ: https://www.kaggle.com/datasets/muhammadkhalid/sign-language-for-alphabets/data




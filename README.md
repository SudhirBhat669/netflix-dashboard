📊 Netflix Data Analysis Dashboard
- An interactive data analysis dashboard using Streamlit, Plotly, and Machine Learning to visualize and explore Netflix titles. The app includes animated charts, genre clustering, a classification model, and PDF export of insights.

📸 Screenshots
<img width="1460" height="1205" alt="Image" src="https://github.com/user-attachments/assets/5a418eb8-9aea-4fc8-9d0b-033d69c63b37" />
<img width="549" height="393" alt="Image" src="https://github.com/user-attachments/assets/8c14a874-522e-4330-b683-f155080889e3" />
<img width="780" height="470" alt="Image" src="https://github.com/user-attachments/assets/8d82a2cc-decc-4a23-b027-7f957b436ba6" />
<img width="859" height="470" alt="Image" src="https://github.com/user-attachments/assets/39381077-d30a-4361-b77a-e802a0bdfbe0" />
<img width="999" height="547" alt="Image" src="https://github.com/user-attachments/assets/429f8a83-465a-450e-9e94-815510c75ca3" />
<img width="797" height="779" alt="Image" src="https://github.com/user-attachments/assets/14d236b8-f1fb-4fe5-93e9-ed1b50f762cb" />
<img width="1143" height="865" alt="Image" src="https://github.com/user-attachments/assets/33ca551e-a370-4e89-be52-74e95ed7ebf4" />

🎯 Objectives
- Visualize Netflix titles by type, genre, country, and release year.
- Detect patterns using animated bar charts and WordClouds.
- Use ML models (Random Forest Classifier) to predict content type.
- Cluster genres using KMeans Clustering.
- Export analysis as PDF.
- Deploy and share via Render or GitHub Pages.

🧰 Tech Stack
- Tool	                 Description
- Python	Core           Programming Language
- Streamlit	       Web Dashboard Framework
- Pandas	               Data Manipulation
- Matplotlib, Seaborn   Static Plots
- Plotly	               Interactive & Animated Charts
- Scikit-learn	       Machine Learning
- ReportLab	       Exporting PDF Reports
- WordCloud	       Word cloud visualization

## 🧾 Folder Structure
netflix-dashboard/
│
├── netflix_analysis.py    # Main Streamlit app
├── netflix1.csv           # Your dataset
├── requirements.txt       # Dependencies
├── README.md              # Project documentation
├── render.yaml            # Render deployment config
└── logs/                  # Logging directory

⚙️ Setup Instructions
🔧 Clone Repository
- https://github.com/SudhirBhat669/netflix-dashboard.git
- cd netflix-dashboard

🐍 Create Virtual Environment
- python -m venv venv
- source venv/bin/activate 
- # on Windows: venv\Scripts\activate

📦 Install Requirements
- pip install -r requirements.txt

▶️ Run the App
- streamlit run netflix_analysis.py

🚀 Features
📊 1. Animated Bar Chart
- Displays the count of Movies/TV Shows by release year using Plotly.

🌍 2. Country & Genre Distribution
- Horizontal and vertical bar charts showing top contributing countries and genres.

☁️ 3. WordCloud
- Generates a WordCloud of frequently used keywords in Netflix titles.

🧠 4. Machine Learning Classification
- Random Forest Classifier trained on duration and rating to predict whether a title is a Movie or TV Show.

🧪 5. KMeans Genre Clustering
- Clusters top 10 genres using KMeans into 3 distinct clusters and plots with Seaborn.

📤 6. Export as PDF
- Exports the entire data insights into a downloadable PDF report using ReportLab.

📌 Future Enhancements
- Add user-based content recommendations using collaborative filtering.
- Use Natural Language Processing to extract sentiment from descriptions.
- Enable multi-page navigation for different visualizations.
- Add filter widgets (release year, genre, country, rating).



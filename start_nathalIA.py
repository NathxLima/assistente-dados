import os
import subprocess
import time

# 1. Fecha possíveis processos na porta 8501
os.system('for /f "tokens=5" %a in (\'netstat -aon ^| find ":8501" ^| find "LISTENING"\') do taskkill /F /PID %a >nul 2>&1')

# 2. Inicia o Streamlit
subprocess.Popen([
    "cmd", "/k",
    "cd /d D:\\PyCharm\\PythonProject\\assistente_dados && streamlit run main.py --server.port 8501 --server.address 0.0.0.0"
])

# Aguarda alguns segundos
time.sleep(3)

# 3. Inicia o Ngrok
subprocess.Popen([
    "cmd", "/k",
    "D:\\ngrok http 8501"
])

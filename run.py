"""Chạy notebook từ đầu đến cuối và lưu kèm kết quả. Dùng trong thư mục notebooks/:

    python3 ../run.py 1_data_understanding_eda.ipynb
"""
import sys
import time
import nbformat
from nbclient import NotebookClient

for path in sys.argv[1:]:
    notebook = nbformat.read(path, 4)
    start = time.time()
    client = NotebookClient(notebook, timeout=7200, kernel_name='python3',
                            resources={'metadata': {'path': '.'}}, allow_errors=False)
    try:
        client.execute()
        status = 'OK'
    except Exception as error:
        status = 'LỖI: ' + str(error)[:300]
    nbformat.write(notebook, path)
    print(status, path, f'{time.time() - start:.0f}s', flush=True)
    if status != 'OK':
        sys.exit(1)

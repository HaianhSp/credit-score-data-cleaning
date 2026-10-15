import nbformat, sys, time
from nbclient import NotebookClient
for f in sys.argv[1:]:
    n = nbformat.read(f, 4); t = time.time()
    c = NotebookClient(n, timeout=7200, kernel_name='python3', resources={'metadata': {'path': '.'}}, allow_errors=False)
    try:
        c.execute(); status = 'OK'
    except Exception as e:
        status = 'ERROR: ' + str(e)[:300]
    nbformat.write(n, f); print(status, f, f'{time.time()-t:.0f}s', flush=True)
    if status != 'OK': sys.exit(1)

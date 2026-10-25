import subprocess,sys

def test_help():
    p=subprocess.run([sys.executable,"-m","rabbitsense","--help"],capture_output=True,text=True)
    assert p.returncode==0, p.stderr
    assert "--version" in p.stdout

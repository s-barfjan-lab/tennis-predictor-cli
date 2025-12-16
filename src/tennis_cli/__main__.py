from .cli import run

if __name__ == "__main__":
    run()

# here when I call python -m tennis_cli, Python executes __main__.py ->__main__.py calls 
# run() from cli.py -> run() calls the Typer app with all commands
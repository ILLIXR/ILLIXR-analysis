import typer

app = typer.Typer()


@app.command()
def main() -> None:
    typer.echo("hello")


# See https://clig.dev/ for guidelines

if __name__ == "__main__":
    app()

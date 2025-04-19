import click
from pathlib import Path
from BlyncsySFT.config import load_and_validate_env
from BlyncsySFT.pipeline import run_auto_training_pipeline


@click.group()
def cli():
    """Training Pipeline CLI"""
    pass


@cli.command()
@click.argument(
    'project_dir',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path)
)
@click.option(
    '--verbose',
    '-v',
    is_flag=True,
    help='Enable verbose output'
)
def train(project_dir, verbose):
    """Run training pipeline with the specified project directory"""

    # Setting required keys
    required = [
        'TRAINING_RUN',
        'EPOCHS',
        'BATCH_SIZE',
        'WORKERS',
        'NUM_CLASSES',
        'BACKBONE',
        'SAVE_EVERY',
        'TRAIN_IMAGE_PATH',
        'TRAIN_ANNOT_PATH',
        'VAL_IMAGE_PATH',
        'VAL_ANNOT_PATH',
    ]

    try:
        # Loading env file
        env_path = (Path(project_dir) / ".env")
        cfg = load_and_validate_env(env_file=env_path, required_keys=required)

        if verbose:
            click.echo("✅ Loaded valid configuration from .env")

        run_auto_training_pipeline(project_dir, cfg, verbose)  # Core logic
        click.echo("🎉 Training completed successfully!")

    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        raise click.Abort()


if __name__ == '__main__':
    cli()

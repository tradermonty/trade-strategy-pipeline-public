"""Smoke tests for package importability."""


def test_package_importable() -> None:
    """import pipeline should succeed and expose __version__."""
    import pipeline

    assert hasattr(pipeline, "__version__")
    assert isinstance(pipeline.__version__, str)
    assert pipeline.__version__ == "0.1.0"

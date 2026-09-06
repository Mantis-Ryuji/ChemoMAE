def test_imports_and_version() -> None:
    import chemomae

    assert hasattr(chemomae, "__version__"), "__version__ missing"
    assert chemomae.__version__ == "0.2.2"


def test_subpackages_visible() -> None:
    import chemomae.preprocessing as P
    import chemomae.models as M
    import chemomae.training as T
    import chemomae.clustering as C
    import chemomae.utils as U

    for mod in (P, M, T, C, U):
        assert mod is not None

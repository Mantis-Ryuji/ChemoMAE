def test_imports_and_version():
    import chemomae
    assert hasattr(chemomae, "__version__"), "__version__ missing"

def test_subpackages_visible():
    import chemomae.preprocessing as P
    import chemomae.models as M
    import chemomae.training as T
    import chemomae.clustering as C
    import chemomae.utils as U
    for mod in (P, M, T, C, U):
        assert mod is not None

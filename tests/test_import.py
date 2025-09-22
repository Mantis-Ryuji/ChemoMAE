def test_imports_and_version():
    import wavemae
    assert hasattr(wavemae, "__version__"), "__version__ missing"

def test_subpackages_visible():
    import wavemae.preprocessing as P
    import wavemae.models as M
    import wavemae.training as T
    import wavemae.clustering as C
    import wavemae.utils as U
    for mod in (P, M, T, C, U):
        assert mod is not None

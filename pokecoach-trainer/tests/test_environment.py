from pokecoach_trainer.upstream import collect_runtime_versions, import_policy_stack


def test_imports_and_versions():
    versions = collect_runtime_versions()
    assert "python" in versions
    import_policy_stack()

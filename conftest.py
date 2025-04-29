import warp as wp


def pytest_addoption(parser):
  parser.addoption(
    "--cpu", action="store_true", default=False, help="run tests with cpu"
  )


def pytest_configure(config):
  if config.getoption("--cpu"):
    wp.set_device("cpu")

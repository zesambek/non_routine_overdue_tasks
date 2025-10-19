"""Custom exception hierarchy for scraper flow."""


class SetupError(RuntimeError):
    ...


class LoginError(RuntimeError):
    ...


class OpenTasksError(RuntimeError):
    ...

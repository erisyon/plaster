"""
To avoid circular references, this class needs to be in a separate file
"""


class ValidationError(Exception):
    # Raise this for any sort of CLI command error
    pass

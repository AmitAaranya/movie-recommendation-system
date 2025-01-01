class MovieNotFoundError(Exception):
    def __init__(self, Id):
        self.message = f"Movie with ID {Id} not found"
        super().__init__(self.message)


class UserNotFoundError(Exception):
    def __init__(self, email):
        self.message = f"User with email {email} not found"
        super().__init__(self.message)
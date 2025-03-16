from datetime import timedelta

class timestamp(timedelta):
    """Timestamp is a wrapper around timedelta with a custom __str__ implementation."""
    def __str__(self):
        """Custom __str__ implementation showing video timestamps (HH:MM:SS)

        Milliseconds and shorter are discarded. Days and longer are converted
        to hours.
        """
        # The core of this logic was copied from the Python 3.12 timedelta lib.
        mm, ss = divmod(self.seconds, 60)
        hh, mm = divmod(mm, 60)
        hh += 24 * self.days
        s = "%02d:%02d:%02d" % (hh, mm, ss)
        return s

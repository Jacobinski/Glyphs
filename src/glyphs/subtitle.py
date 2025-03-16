import srt

from nltk import edit_distance
from glyphs.timestamp import timestamp

class SubtitleGenerator:
    """Generates SRT formatted subtitles."""
    subtitles = []
    current_content = ""
    current_start_timestamp = None
    current_recent_timestamp = None
    _index = 1  # Access with self.current_index()
    __verbose: bool

    def __init__(self, verbose: bool = False):
        self.__verbose = verbose

    def current_index(self):
        i = self._index
        self._index += 1
        return i

    # TODO: We can implement outlier detection by maintaining a count of each repeated subtitle
    #       and pruning (or merging) subtitles with low repetition in the create_srt() function.
    def add_subtitle(self, time: timestamp, content: str):
        remove_punctuation = str.maketrans({
            '，': '',
            '.': '',
            '。': '',
            '．': '',
            '、': '',
        })
        self.current_recent_timestamp = time
        if self.current_content.translate(remove_punctuation) == content.translate(remove_punctuation):
            if self.__verbose:
                print(f"  MATCH: \"{content}\" == \"{self.current_content}\"")
        else:
            if self.current_content == "":
                if self.__verbose:
                    print(f"  INIT: \"{content}\"")
                self.current_content = content
                self.current_start_timestamp = time
            else:
                if self.__verbose:
                    print(f"  OVERWRITE: \"{self.current_content}\" -> \"{content}\" (distance: {edit_distance(self.current_content, content)})")
                self.subtitles.append(
                    srt.Subtitle(
                        index = self.current_index(),
                        start = self.current_start_timestamp,
                        end = self.current_recent_timestamp,
                        content = self.current_content,
                    )
                )
                self.current_start_timestamp = time
                self.current_content = content

    def create_srt(self):
        if self.current_content != "":
            start = self.current_start_timestamp
            end = self.current_recent_timestamp
            content = self.current_content
            self.subtitles.append(
                srt.Subtitle(
                    index = self.current_index(),
                    start = start,
                    end = end,
                    content = content,
                )
            )
            print(f"[{start}-{end}] {content}")
            self.current_content = ""
            self.current_recent_timestamp = None
            self.current_start_timestamp = None
        return srt.compose(self.subtitles)

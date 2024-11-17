import unittest
import os

from main import process_video

class IntegerArithmeticTestCase(unittest.TestCase):
    def testAdd(self):  # test method names begin with 'test'
        self.assertEqual((1 + 2), 3)
        self.assertEqual(0 + 1, 1)
    def testMultiply(self):
        self.assertEqual((0 * 10), 0)
        self.assertEqual((5 * 8), 40)

class SubtitleGenerationTestCase(unittest.TestCase):
    maxDiff = None
    def testSubtitleGeneration(self):
        want = """1
00:00:00,600 --> 00:00:02,480
大家快看，那是什么

2
00:00:05,680 --> 00:00:07,800
飞得那么慢又长得这么奇怪，是虫子吧

3
00:00:12,640 --> 00:00:12,800
是村长

4
00:00:14,160 --> 00:00:14,200
对了，刚才是谁说我新发明的

5
00:00:14,200 --> 00:00:17,040
寸了，刚才是谁说我新发明的

6
00:00:17,040 --> 00:00:17,840
对了，刚才是谁说我新发明的

7
00:00:18,040 --> 00:00:21,360
极速视频通话器飞得慢了

8
00:00:25,560 --> 00:00:26,760
这个机器，一分钟能够前进一百米

9
00:00:28,400 --> 00:00:29,280
实在是方便得很啊

"""
        file = "test_clips/ep5_30s_to_60s.mkv"
        subtitles = process_video(file)

        # Write to file for debugging
        srt_file = os.path.splitext(file)[0] + ".srt"
        with open(srt_file, "w", encoding='utf-8') as f:
            f.write(subtitles)

        self.assertEqual(subtitles, want)


if __name__ == '__main__':
    unittest.main()
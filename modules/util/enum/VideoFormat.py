from enum import Enum


class VideoFormat(Enum):
    PNG_IMAGE_SEQUENCE = 'PNG_IMAGE_SEQUENCE'
    JPG_IMAGE_SEQUENCE = 'JPG_IMAGE_SEQUENCE'
    MP4 = 'MP4'
    WEBM = 'WEBM'
    AVI = 'AVI'

    def __str__(self):
        return self.value

    def extension(self) -> str:
        match self:
            case VideoFormat.PNG_IMAGE_SEQUENCE:
                return ".png"
            case VideoFormat.JPG_IMAGE_SEQUENCE:
                return ".jpg"
            case VideoFormat.MP4:
                return ".mp4"
            case VideoFormat.WEBM:
                return ".webm"
            case VideoFormat.AVI:
                return ".avi"
            case _:
                return ""

    def pil_format(self) -> str:
        match self:
            case VideoFormat.PNG_IMAGE_SEQUENCE:
                return "PNG"
            case VideoFormat.JPG_IMAGE_SEQUENCE:
                return "JPEG"
            case _:
                return ""

    def is_video_format(self) -> bool:
        """Check if this format represents an actual video file (not image sequence)"""
        return self in [VideoFormat.MP4, VideoFormat.WEBM, VideoFormat.AVI]

    def codec_options(self) -> dict:
        """Get codec options for video encoding"""
        match self:
            case VideoFormat.MP4:
                return {"codec": "libx264", "crf": "17", "preset": "medium"}
            case VideoFormat.WEBM:
                return {"codec": "libvpx-vp9", "crf": "30", "preset": "medium"}
            case VideoFormat.AVI:
                return {"codec": "libx264", "crf": "17", "preset": "medium"}
            case _:
                return {}

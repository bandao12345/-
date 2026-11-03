import base64
import io
import unittest
from PIL import Image, ImageDraw
from image_io import (
    ImageValidationError, decode_base64_image, make_preview,
    prepare_image, validate_image_bytes,
)


def encoded(image, format="PNG", **options):
    buffer=io.BytesIO()
    image.save(buffer,format=format,**options)
    return buffer.getvalue()


class ImageIOTests(unittest.TestCase):
    def test_real_png_and_strict_base64_preserve_dimensions(self):
        image=Image.new("RGB",(300,100),"white")
        ImageDraw.Draw(image).line((40,30,180,70),fill="black",width=8)
        raw=encoded(image)
        self.assertEqual(validate_image_bytes(raw).size,(300,100))
        self.assertEqual(decode_base64_image("data:image/png;base64,"+base64.b64encode(raw).decode()).size,(300,100))
        with self.assertRaises(ImageValidationError):
            decode_base64_image(base64.b64encode(raw).decode()[:30]+"!"+base64.b64encode(raw).decode()[30:])

    def test_invalid_types_formats_and_corruption_are_rejected(self):
        for value in (None,1,[],{},True,"not an image"):
            with self.subTest(value=value):
                with self.assertRaises(ImageValidationError):
                    decode_base64_image(value)
        for value in (b"",b"not an image",encoded(Image.new("L",(20,20)),"BMP")):
            with self.assertRaises(ImageValidationError):
                validate_image_bytes(value)

    def test_blank_and_transparent_blank_are_rejected(self):
        for image in (Image.new("RGB",(100,50),color) for color in ("white","black","gray")):
            with self.assertRaises(ImageValidationError):
                prepare_image(image)
        with self.assertRaises(ImageValidationError):
            prepare_image(Image.new("RGBA",(100,50),(255,255,255,0)))

    def test_transparent_black_or_white_ink_are_both_visible(self):
        for ink in ((0,0,0,255),(255,255,255,255)):
            image=Image.new("RGBA",(100,80),(0,0,0,0))
            ImageDraw.Draw(image).line((30,20,70,60),fill=ink,width=7)
            normalized=prepare_image(image)
            self.assertEqual(normalized.mode,"RGB")
            self.assertGreater(normalized.getpixel((0,0))[0],200)
            self.assertLess(normalized.getpixel((50,40))[0],40)

    def test_limits_and_exif_orientation(self):
        with self.assertRaises(ImageValidationError):
            validate_image_bytes(encoded(Image.new("L",(4097,10),255)))
        with self.assertRaises(ImageValidationError):
            validate_image_bytes(b"x"*(2*1024*1024+1))
        image=Image.new("RGB",(100,60),"white")
        ImageDraw.Draw(image).rectangle((10,10,40,40),fill="black")
        exif=Image.Exif()
        exif[274]=6
        oriented=prepare_image(validate_image_bytes(encoded(image,"JPEG",exif=exif)))
        self.assertEqual(oriented.size,(60,100))

    def test_preview_preserves_layout_and_has_bounded_dimensions(self):
        image=Image.new("RGB",(1200,600),"white")
        ImageDraw.Draw(image).line((80,100,450,400),fill="black",width=14)
        value=make_preview(image)
        preview=decode_base64_image(value)
        self.assertEqual(preview.size,(1000,500))


if __name__=="__main__":
    unittest.main()


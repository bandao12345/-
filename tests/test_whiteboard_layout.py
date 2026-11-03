"""Geometry checks only. These tests make no OCR accuracy claim."""
import unittest
from PIL import Image,ImageDraw
from whiteboard_layout import line_regions

class GeometryBehaviour(unittest.TestCase):
    def test_detached_latin_dot_stays_with_stem(self):
        im=Image.new('RGB',(600,400),'white');d=ImageDraw.Draw(im)
        d.ellipse((100,80,112,92),fill='black');d.line((106,115,106,250),fill='black',width=4)
        regions=line_regions(im)
        self.assertEqual(len(regions),1)
        self.assertLess(regions[0][1],80);self.assertGreater(regions[0][3],250)

    def test_upper_and_lower_han_strokes_not_split_into_rows(self):
        im=Image.new('RGB',(600,400),'white');d=ImageDraw.Draw(im)
        for y in (100,145,190):d.line((150,y,290,y),fill='black',width=4)
        self.assertEqual(len(line_regions(im)),1)

    def test_two_rows_keep_reading_order(self):
        im=Image.new('RGB',(800,600),'white');d=ImageDraw.Draw(im)
        for x,y in ((100,50),(250,50),(70,350),(260,350),(440,350)):
            d.line((x,y,x,y+120),fill='black',width=4)
            d.line((x,y,x+80,y),fill='black',width=4)
        regions=line_regions(im)
        self.assertEqual(len(regions),2)
        self.assertLess(regions[0][3],regions[1][1])

    def test_light_saturated_colors_keep_same_layout(self):
        for color in ('yellow','red','cyan',(230,160,30)):
            with self.subTest(color=color):
                im=Image.new('RGB',(600,400),'white');d=ImageDraw.Draw(im)
                d.line((100,100,100,260),fill=color,width=4)
                d.line((100,100,200,100),fill=color,width=4)
                self.assertEqual(len(line_regions(im)),1)

    def test_non_white_photo_background_routes_to_detector(self):
        self.assertIsNone(line_regions(Image.new('RGB',(600,400),(180,180,180))))

    def test_border_writing_coordinates_are_bounded(self):
        im=Image.new('RGB',(300,200),'white');d=ImageDraw.Draw(im)
        d.line((0,0,0,199),fill='black',width=6);d.line((0,0,50,0),fill='black',width=6)
        for a,b,c,e in line_regions(im):
            self.assertGreaterEqual(a,0);self.assertGreaterEqual(b,0)
            self.assertLessEqual(c,300);self.assertLessEqual(e,200)

if __name__=='__main__':unittest.main()

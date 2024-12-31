from artnet import Scene, RGB
import math

white = RGB(255, 255, 255)
red = RGB(255, 0, 0)
blue = RGB(0, 0, 255)
green = RGB(0, 255, 0)
black = RGB(0, 0, 0)

class CalibrationScene(Scene):
    def render(self, raster, time):
        for y in range(raster.height):
            for x in range(raster.width):
                for z in range(raster.length):
                    idx = y * raster.width + x + z * raster.width * raster.height

                    
                    color = black
                    if x == 0:
                        color = red
                    elif y == 0:
                        color = green
                    elif z == 0:
                        color = blue
                    
                    raster.data[idx] = color
                    
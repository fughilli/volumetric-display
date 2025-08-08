from artnet import RGB, Scene


# Blink each of the layers on and off in sequence.
class FullWhiteScene(Scene):

    def render(self, raster, time):
        for y in range(raster.height):
            for x in range(raster.width):
                for z in range(raster.length):
                    idx = y * raster.width + x + z * raster.width * raster.height
                    dot_on = int(idx + time * 30) % 4 == 0
                    raster.data[idx] = RGB(255, 255, 255) if dot_on else RGB(0, 0, 0)

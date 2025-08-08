from artnet import RGB, Scene


# Blink each of the layers on and off in sequence.
class EnumerateScene(Scene):

    def __init__(self):
        self.index = 0

    def render(self, raster, time):
        if input("Press `n` to advance to the next layer: ").strip() == "n":
            self.index += 1
            self.index = self.index % raster.length

        print("Drawing layer %d" % self.index)

        for y in range(raster.height):
            for x in range(raster.width):
                for z in range(raster.length):
                    idx = y * raster.width + x + z * raster.width * raster.height
                    layer_on = self.index % raster.length == z
                    raster.data[idx] = RGB(255, 255, 255) if layer_on else RGB(0, 0, 0)

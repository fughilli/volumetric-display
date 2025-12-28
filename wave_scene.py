import numpy as np

from artnet import (  # RGB and HSV are no longer needed for this implementation
    Raster,
    Scene,
)


class WaveScene(Scene):
    def __init__(self, **kwargs):
        pass

    def render(self, raster: Raster, time: float):
        raster.data.fill(0)

        ht = raster.height

        frequency = 2 * np.pi / 80
        for x in range(raster.width):
            for z in range(raster.length):
                # Normalize x and z to a range [0, 2π] for smooth sine behavior
                xf = x * frequency
                zf = z * frequency

                # adjust speed
                ts = time / 2
                # simplest possible y=sin(x+t)
                # y = int((np.sin(xf + ts*2) * 20) + (raster.height // 2))

                # vary by x and z
                # y = int((np.sin(xf + zf + ts) * 20) + (raster.height // 2))

                # change the wave shape over time
                y = int(
                    (np.sin(xf * np.cos(ts) * 4 + zf * np.sin(ts) + ts) * 20) + (raster.height // 2)
                )

                # only minor wave shape changes over time
                # y = int((np.sin(xf*5 + zf + 2*time + np.sin(ts)*zf) * 20) + (raster.height // 2))

                # Clamp y to grid bounds
                if 0 <= y < raster.height:
                    # some random variations in color with time + space
                    raster.data[x, y, z, :] = [
                        128 + 110 * np.sin(x / ht * time + y / ht * 1.5 + time),
                        128 + 110 * np.sin(z / ht * 2 + y / ht + time * 2 + np.pi / 2),
                        128 + 110 * np.sin(x / ht + z / ht + time * 3 + np.pi),
                    ]

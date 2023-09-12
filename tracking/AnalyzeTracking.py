import sys
from pathlib import Path
sys.path.append(str(Path(".").resolve()))
from .ImageHandling import LoadPILImages, GetFrames
import numpy as np
import skimage.measure
import skimage.morphology
import skimage.filters
from PIL import ImageFont, ImageDraw, Image

import imageio
imageio.plugins.freeimage.download()

def DrawAndHighlight(labeled, image, labelsAndColorToHighlight, savePath):
    overlay = Image.new("RGBA", tuple(reversed(image.shape)), (255, 255, 255, 0))
    drawer = ImageDraw.Draw(overlay)
    font = ImageFont.truetype("arial.ttf", 24)
    for rp in skimage.measure.regionprops(labeled):
        if rp.label in labelsAndColorToHighlight:
            color = labelsAndColorToHighlight[rp.label]
        else:
            color = (255, 255, 255)
        drawer.point(list(rp.coords[:, (1, 0)].flatten()), color + tuple([100]))
        outline = skimage.morphology.binary_dilation(
            skimage.filters.sobel(rp.image, mode='constant') > 0)
        outlineCoords = np.argwhere(outline) + rp.bbox[:2]
        drawer.point(list(outlineCoords[:, (1, 0)].flatten()), color + tuple([100]))
        xC, yC = reversed(rp.centroid)
        drawer.text((xC, yC), str(rp.label), anchor="ms", fill=(255, 255, 255, 255), font=font)
    image = Image.fromarray(image).convert("RGBA")
    image = Image.alpha_composite(image, overlay)
    image.save(savePath)



